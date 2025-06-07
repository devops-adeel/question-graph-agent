"""
Integration module for connecting memory retrieval with graph nodes.

This module provides utilities and enhanced nodes that integrate memory
retrieval capabilities into the question-answer graph flow.
"""

import logging
from typing import Optional, Dict, Any, List
from datetime import datetime

import logfire
from pydantic import Field
from pydantic_graph import BaseNode, GraphRunContext

from question_graph import QuestionState, Answer
from graphiti_client import GraphitiClient
from memory_retrieval import MemoryRetrieval
from enhanced_question_agent import EnhancedQuestionAgent, QuestionGenerationContext
from graphiti_memory import MemoryStorage, QAPair
from graphiti_entities import DifficultyLevel


logger = logging.getLogger(__name__)


class MemoryEnhancedAsk(BaseNode[QuestionState]):
    """Ask node enhanced with memory retrieval for intelligent question generation."""
    
    async def run(self, ctx: GraphRunContext[QuestionState]) -> Answer:
        with logfire.span('memory_enhanced_ask.generate_question') as span:
            span.set_attribute('ask_agent_messages_count', len(ctx.state.ask_agent_messages))
            
            # Check if we have GraphitiClient
            if hasattr(ctx.state, 'graphiti_client') and ctx.state.graphiti_client:
                graphiti_client = ctx.state.graphiti_client
                span.set_attribute('memory_enabled', True)
                
                # Get user and session info
                user_id = ctx.state.current_user.id if hasattr(ctx.state, 'current_user') and ctx.state.current_user else "default_user"
                session_id = ctx.state.session_id if hasattr(ctx.state, 'session_id') else None
                
                span.set_attribute('user_id', user_id)
                span.set_attribute('session_id', session_id or "no_session")
                
                # Use enhanced question agent
                agent = EnhancedQuestionAgent(graphiti_client=graphiti_client)
                
                # Build context for question generation
                context = QuestionGenerationContext(
                    user_id=user_id,
                    session_id=session_id,
                    prefer_weak_topics=True
                )
                
                # Check if we should adjust difficulty based on recent performance
                if hasattr(ctx.state, 'consecutive_incorrect') and ctx.state.consecutive_incorrect >= 2:
                    context.difficulty_override = DifficultyLevel.EASY
                    span.set_attribute('difficulty_adjusted', True)
                
                try:
                    # Generate question with memory context
                    question = await agent.generate_question(
                        context=context,
                        message_history=ctx.state.ask_agent_messages
                    )
                    
                    ctx.state.question = question
                    span.set_attribute('generated_question', question)
                    span.set_attribute('generation_method', 'memory_enhanced')
                    
                    # Store the question if we have memory storage
                    memory_storage = MemoryStorage(client=graphiti_client)
                    question_entity = await memory_storage.store_question_only(
                        question=question,
                        difficulty=context.difficulty_override or DifficultyLevel.MEDIUM
                    )
                    
                    if question_entity:
                        if hasattr(ctx.state, 'current_question_id'):
                            ctx.state.current_question_id = question_entity.id
                        span.set_attribute('question_stored', True)
                        span.set_attribute('question_id', question_entity.id)
                    
                    logfire.info('Memory-enhanced question generated', 
                               question=question,
                               user_id=user_id,
                               method='memory_enhanced')
                    
                except Exception as e:
                    logger.error(f"Memory-enhanced generation failed: {e}")
                    span.set_attribute('memory_error', str(e))
                    # Fall back to standard generation
                    question = await self._fallback_generation(ctx)
                    ctx.state.question = question
                    
            else:
                # No memory available, use standard generation
                span.set_attribute('memory_enabled', False)
                question = await self._fallback_generation(ctx)
                ctx.state.question = question
            
            return Answer(ctx.state.question)
    
    async def _fallback_generation(self, ctx: GraphRunContext[QuestionState]) -> str:
        """Fallback to standard question generation."""
        from pydantic_ai import Agent
        
        agent = Agent('openai:gpt-4o', output_type=str)
        result = await agent.run(
            'Ask a simple question with a single correct answer.',
            message_history=ctx.state.ask_agent_messages,
        )
        ctx.state.ask_agent_messages += result.all_messages()
        
        logfire.info('Fallback question generation used')
        return result.output


class MemoryContextBuilder:
    """Builds context for memory-enhanced operations."""
    
    def __init__(self, graphiti_client: Optional[GraphitiClient] = None):
        """Initialize context builder.
        
        Args:
            graphiti_client: Graphiti client for memory access
        """
        self.graphiti_client = graphiti_client
        self.retrieval = MemoryRetrieval(client=graphiti_client) if graphiti_client else None
    
    async def build_question_context(self,
                                     user_id: str,
                                     session_id: Optional[str] = None) -> Dict[str, Any]:
        """Build context for question generation.
        
        Args:
            user_id: User ID
            session_id: Optional session ID
            
        Returns:
            Context dictionary
        """
        context = {
            "user_id": user_id,
            "session_id": session_id,
            "timestamp": datetime.now()
        }
        
        if not self.retrieval:
            return context
        
        try:
            # Add performance data
            performance = await self.retrieval.get_user_performance(user_id)
            context["user_performance"] = performance
            
            # Add weak topics
            weak_topics = await self.retrieval.get_weak_topics(user_id)
            context["weak_topics"] = weak_topics
            
            # Add recent question count
            recent_questions = await self.retrieval.get_asked_questions(
                user_id, 
                session_id,
                limit=10
            )
            context["recent_question_count"] = len(recent_questions)
            
        except Exception as e:
            logger.error(f"Failed to build question context: {e}")
        
        return context
    
    async def build_evaluation_context(self,
                                       question_id: str,
                                       user_id: str) -> Dict[str, Any]:
        """Build context for answer evaluation.
        
        Args:
            question_id: Question ID
            user_id: User ID
            
        Returns:
            Context dictionary
        """
        context = {
            "question_id": question_id,
            "user_id": user_id,
            "timestamp": datetime.now()
        }
        
        if not self.retrieval:
            return context
        
        try:
            # Add question context
            question_context = await self.retrieval.get_question_context(question_id)
            context["question_context"] = question_context
            
            # Add user's performance on related topics
            if "topics" in question_context:
                topic_performance = {}
                for topic in question_context["topics"]:
                    perf = await self.retrieval.get_topic_performance(user_id, topic)
                    topic_performance[topic] = perf
                context["topic_performance"] = topic_performance
            
        except Exception as e:
            logger.error(f"Failed to build evaluation context: {e}")
        
        return context


class MemoryStateEnhancer:
    """Enhances QuestionState with memory-related fields and methods."""
    
    @staticmethod
    def enhance_state(state: QuestionState,
                      graphiti_client: Optional[GraphitiClient] = None,
                      user_id: Optional[str] = None,
                      session_id: Optional[str] = None) -> QuestionState:
        """Enhance a QuestionState with memory capabilities.
        
        Args:
            state: Base QuestionState
            graphiti_client: Optional Graphiti client
            user_id: Optional user ID
            session_id: Optional session ID
            
        Returns:
            Enhanced QuestionState
        """
        # Add GraphitiClient if provided
        if graphiti_client and not hasattr(state, 'graphiti_client'):
            state.graphiti_client = graphiti_client
        
        # Add user info
        if user_id and not hasattr(state, 'current_user'):
            from graphiti_entities import UserEntity
            state.current_user = UserEntity(
                id=user_id,
                session_id=session_id or f"session_{datetime.now().timestamp()}",
                total_questions=0,
                correct_answers=0,
                average_response_time=0.0
            )
        
        # Add session ID
        if session_id and not hasattr(state, 'session_id'):
            state.session_id = session_id
        
        # Add tracking fields
        if not hasattr(state, 'current_question_id'):
            state.current_question_id = None
        
        if not hasattr(state, 'last_response_time'):
            state.last_response_time = None
        
        if not hasattr(state, 'consecutive_incorrect'):
            state.consecutive_incorrect = 0
        
        if not hasattr(state, 'question_count'):
            state.question_count = 0
        
        return state


async def create_memory_enhanced_graph():
    """Create a question graph with memory enhancement.
    
    Returns:
        Enhanced graph instance
    """
    from pydantic_graph import Graph
    from enhanced_nodes import EnhancedAnswer, EnhancedEvaluate, EnhancedReprimand
    
    # Use memory-enhanced Ask node
    nodes = (
        MemoryEnhancedAsk,
        EnhancedAnswer,
        EnhancedEvaluate,
        EnhancedReprimand
    )
    
    graph = Graph(
        nodes=nodes,
        state_type=QuestionState
    )
    
    return graph