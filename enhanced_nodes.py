"""
Enhanced graph nodes with memory storage integration.

This module provides enhanced versions of the graph nodes that integrate
with the memory storage system to persist Q&A interactions.
"""

import time
import logging
from typing import Optional
from datetime import datetime

import logfire
from pydantic import Field
from pydantic_graph import BaseNode, End, GraphRunContext

from question_graph import (
    QuestionState,
    Answer,
    Evaluate,
    Reprimand,
    EvaluationOutput,
    ask_agent,
    evaluate_agent,
)
from graphiti_memory import MemoryStorage, QAPair, DifficultyLevel
from graphiti_entities import QuestionEntity
from entity_extraction import EntityExtractor


logger = logging.getLogger(__name__)


class EnhancedAsk(BaseNode[QuestionState]):
    """Enhanced Ask node that stores questions in memory."""
    
    async def run(self, ctx: GraphRunContext[QuestionState]) -> Answer:
        with logfire.span('enhanced_ask_node.generate_question') as span:
            span.set_attribute('ask_agent_messages_count', len(ctx.state.ask_agent_messages))
            
            # Check if we have GraphitiClient and MemoryStorage
            memory_storage = None
            if hasattr(ctx.state, 'graphiti_client') and ctx.state.graphiti_client:
                memory_storage = MemoryStorage(client=ctx.state.graphiti_client)
                span.set_attribute('memory_enabled', True)
            else:
                span.set_attribute('memory_enabled', False)
            
            # Generate question
            result = await ask_agent.run(
                'Ask a simple question with a single correct answer.',
                message_history=ctx.state.ask_agent_messages,
            )
            ctx.state.ask_agent_messages += result.all_messages()
            ctx.state.question = result.output
            
            span.set_attribute('generated_question', result.output)
            logfire.info('Question generated', question=result.output)
            
            # Store question in memory if available
            if memory_storage:
                try:
                    question_entity = await memory_storage.store_question_only(
                        question=result.output,
                        difficulty=DifficultyLevel.MEDIUM  # Could be determined by AI
                    )
                    if question_entity:
                        # Store question ID in state for later use
                        if hasattr(ctx.state, 'current_question_id'):
                            ctx.state.current_question_id = question_entity.id
                        span.set_attribute('question_stored', True)
                        span.set_attribute('question_id', question_entity.id)
                        logfire.info('Question stored in memory', question_id=question_entity.id)
                except Exception as e:
                    logger.error(f"Failed to store question in memory: {e}")
                    span.set_attribute('memory_error', str(e))
            
            return Answer(result.output)


class EnhancedAnswer(BaseNode[QuestionState]):
    """Enhanced Answer node that tracks response time."""
    
    question: str = Field(
        min_length=1,
        description="The question to ask the user"
    )
    
    async def run(self, ctx: GraphRunContext[QuestionState]) -> Evaluate:
        with logfire.span('enhanced_answer_node.collect_user_input') as span:
            span.set_attribute('question', self.question)
            
            # Track response time
            start_time = time.time()
            answer = input(f'{self.question}: ')
            response_time = time.time() - start_time
            
            span.set_attribute('user_answer', answer)
            span.set_attribute('response_time', response_time)
            logfire.info('User answer collected', 
                        question=self.question, 
                        answer=answer,
                        response_time=response_time)
            
            # Store response time in state for later use
            if hasattr(ctx.state, 'last_response_time'):
                ctx.state.last_response_time = response_time
            
            return EnhancedEvaluate(answer, response_time=response_time)


class EnhancedEvaluate(BaseNode[QuestionState, None, str]):
    """Enhanced Evaluate node that stores complete Q&A pairs."""
    
    answer: str = Field(
        min_length=1,
        description="The user's answer to evaluate"
    )
    response_time: Optional[float] = Field(
        default=None,
        description="Time taken to answer in seconds"
    )
    
    async def run(
        self,
        ctx: GraphRunContext[QuestionState],
    ) -> End[str] | Reprimand:
        with logfire.span('enhanced_evaluate_node.assess_answer') as span:
            if ctx.state.question is None:
                logfire.error('No question available for evaluation')
                raise ValueError("No question available to evaluate against")
            
            span.set_attribute('question', ctx.state.question)
            span.set_attribute('user_answer', self.answer)
            span.set_attribute('response_time', self.response_time)
            
            # Evaluate answer
            result = await evaluate_agent.run(
                format_as_xml({'question': ctx.state.question, 'answer': self.answer}),
                message_history=ctx.state.evaluate_agent_messages,
            )
            ctx.state.evaluate_agent_messages += result.all_messages()
            
            span.set_attribute('evaluation_correct', result.output.correct)
            span.set_attribute('evaluation_comment', result.output.comment)
            
            # Store complete Q&A pair in memory
            if hasattr(ctx.state, 'graphiti_client') and ctx.state.graphiti_client:
                memory_storage = MemoryStorage(client=ctx.state.graphiti_client)
                
                try:
                    # Create QA pair
                    qa_pair = QAPair(
                        question=ctx.state.question,
                        answer=self.answer,
                        user_id=ctx.state.current_user.id if hasattr(ctx.state, 'current_user') and ctx.state.current_user else "unknown",
                        session_id=ctx.state.session_id if hasattr(ctx.state, 'session_id') else "unknown",
                        correct=result.output.correct,
                        evaluation_comment=result.output.comment,
                        response_time=self.response_time,
                        confidence_score=0.8 if result.output.correct else 0.3  # Simple heuristic
                    )
                    
                    # If we stored the question earlier, use that ID
                    if hasattr(ctx.state, 'current_question_id'):
                        qa_pair.question_id = ctx.state.current_question_id
                    
                    # Store the complete Q&A pair
                    success = await memory_storage.store_qa_pair(qa_pair)
                    
                    span.set_attribute('qa_pair_stored', success)
                    if success:
                        logfire.info('Q&A pair stored in memory', 
                                   question_id=qa_pair.question_id,
                                   answer_id=qa_pair.answer_id,
                                   correct=result.output.correct)
                    
                except Exception as e:
                    logger.error(f"Failed to store Q&A pair: {e}")
                    span.set_attribute('memory_error', str(e))
            
            if result.output.correct:
                logfire.info('Answer evaluated as correct', 
                           question=ctx.state.question, 
                           answer=self.answer, 
                           comment=result.output.comment)
                return End(result.output.comment)
            else:
                logfire.warning('Answer evaluated as incorrect', 
                              question=ctx.state.question, 
                              answer=self.answer, 
                              comment=result.output.comment)
                return EnhancedReprimand(result.output.comment)


class EnhancedReprimand(BaseNode[QuestionState]):
    """Enhanced Reprimand node that tracks incorrect answers."""
    
    comment: str = Field(
        min_length=1,
        description="Feedback comment for the incorrect answer"
    )
    
    async def run(self, ctx: GraphRunContext[QuestionState]) -> EnhancedAsk:
        with logfire.span('enhanced_reprimand_node.provide_feedback') as span:
            span.set_attribute('feedback_comment', self.comment)
            
            print(f'Comment: {self.comment}')
            logfire.info('Providing feedback for incorrect answer', comment=self.comment)
            
            # Track consecutive incorrect answers if available
            if hasattr(ctx.state, 'consecutive_incorrect'):
                ctx.state.consecutive_incorrect = getattr(ctx.state, 'consecutive_incorrect', 0) + 1
                span.set_attribute('consecutive_incorrect', ctx.state.consecutive_incorrect)
                
                # Could adjust difficulty based on performance
                if ctx.state.consecutive_incorrect >= 3:
                    logfire.warning('User struggling with questions', 
                                  consecutive_incorrect=ctx.state.consecutive_incorrect)
            
            ctx.state.question = None
            return EnhancedAsk()


def create_enhanced_state(base_state: Optional[QuestionState] = None) -> QuestionState:
    """Create an enhanced QuestionState with additional fields for memory tracking.
    
    Args:
        base_state: Base state to enhance
        
    Returns:
        Enhanced QuestionState
    """
    if base_state is None:
        base_state = QuestionState()
    
    # Add additional tracking fields
    # Note: These would ideally be part of an EnhancedQuestionState class
    # but for compatibility we're adding them dynamically
    if not hasattr(base_state, 'current_question_id'):
        base_state.current_question_id = None
    
    if not hasattr(base_state, 'last_response_time'):
        base_state.last_response_time = None
    
    if not hasattr(base_state, 'consecutive_incorrect'):
        base_state.consecutive_incorrect = 0
    
    return base_state