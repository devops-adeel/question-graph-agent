from __future__ import annotations as _annotations

from pathlib import Path
from typing import List, Optional

import logfire
from pydantic import BaseModel, Field, field_validator, ValidationError
from pydantic_graph import (
    BaseNode,
    End,
    Graph,
    GraphRunContext,
)
from pydantic_graph.persistence.file import FileStatePersistence

from pydantic_ai import Agent, format_as_xml
from pydantic_ai.messages import ModelMessage

from graphiti_client import GraphitiClient
from graphiti_entities import UserEntity

# Import for type checking to avoid circular imports
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from graphiti_client import GraphitiClient as _GraphitiClient

# 'if-token-present' means nothing will be sent (and the example will work) if you don't have logfire configured
logfire.configure(send_to_logfire='if-token-present')
logfire.instrument_pydantic_ai()

ask_agent = Agent('openai:gpt-4o', output_type=str)


class QuestionState(BaseModel):
    """State container for the question-answer conversation flow.
    
    Maintains conversation state including current question and agent message histories.
    Compatible with pydantic_graph's FileStatePersistence for session restoration.
    Now includes GraphitiClient for temporal knowledge graph integration.
    """
    model_config = {"arbitrary_types_allowed": True}
    
    question: str | None = Field(
        default=None,
        description="The current question being asked to the user"
    )
    ask_agent_messages: List[ModelMessage] = Field(
        default_factory=list,
        description="Message history for the question-asking agent",
        exclude_unset=True
    )
    evaluate_agent_messages: List[ModelMessage] = Field(
        default_factory=list,
        description="Message history for the answer evaluation agent", 
        exclude_unset=True
    )
    graphiti_client: Optional[GraphitiClient] = Field(
        default=None,
        description="Graphiti client for knowledge graph interactions",
        exclude=True  # Exclude from serialization
    )
    current_user: Optional[UserEntity] = Field(
        default=None,
        description="Current user entity for the session",
        exclude=True  # Exclude from serialization
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Session ID for tracking Q&A interactions"
    )


class Ask(BaseNode[QuestionState]):
    """Node that generates questions using the ask agent."""
    
    async def run(self, ctx: GraphRunContext[QuestionState]) -> Answer:
        with logfire.span('ask_node.generate_question') as span:
            span.set_attribute('ask_agent_messages_count', len(ctx.state.ask_agent_messages))
            
            result = await ask_agent.run(
                'Ask a simple question with a single correct answer.',
                message_history=ctx.state.ask_agent_messages,
            )
            ctx.state.ask_agent_messages += result.all_messages()
            ctx.state.question = result.output
            
            span.set_attribute('generated_question', result.output)
            logfire.info('Question generated', question=result.output)
            
            return Answer(question=result.output)


class Answer(BaseNode[QuestionState]):
    """Node that prompts user for input and captures their answer."""
    
    question: str = Field(
        min_length=1,
        description="The question to ask the user"
    )
    
    @field_validator('question')
    @classmethod
    def validate_question_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Question cannot be empty or whitespace only")
        return v.strip()

    async def run(self, ctx: GraphRunContext[QuestionState]) -> Evaluate:
        with logfire.span('answer_node.collect_user_input') as span:
            span.set_attribute('question', self.question)
            
            answer = input(f'{self.question}: ')
            
            span.set_attribute('user_answer', answer)
            logfire.info('User answer collected', question=self.question, answer=answer)
            
            return Evaluate(answer=answer)


class EvaluationOutput(BaseModel, use_attribute_docstrings=True):
    correct: bool
    """Whether the answer is correct."""
    comment: str
    """Comment on the answer, reprimand the user if the answer is wrong."""


evaluate_agent = Agent(
    'openai:gpt-4o',
    output_type=EvaluationOutput,
    system_prompt='Given a question and answer, evaluate if the answer is correct.',
)


class Evaluate(BaseNode[QuestionState, None, str]):
    """Node that evaluates user answers for correctness."""
    
    answer: str = Field(
        min_length=1,
        description="The user's answer to evaluate"
    )
    
    @field_validator('answer')
    @classmethod
    def validate_answer_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Answer cannot be empty or whitespace only")
        return v.strip()

    async def run(
        self,
        ctx: GraphRunContext[QuestionState],
    ) -> End[str] | Reprimand:
        with logfire.span('evaluate_node.assess_answer') as span:
            if ctx.state.question is None:
                logfire.error('No question available for evaluation')
                raise ValidationError.from_exception_data(
                    "Evaluate", 
                    [{"type": "missing", "loc": ("question",), "msg": "No question available to evaluate against", "input": None}]
                )
            
            span.set_attribute('question', ctx.state.question)
            span.set_attribute('user_answer', self.answer)
            span.set_attribute('evaluate_agent_messages_count', len(ctx.state.evaluate_agent_messages))
            
            result = await evaluate_agent.run(
                format_as_xml({'question': ctx.state.question, 'answer': self.answer}),
                message_history=ctx.state.evaluate_agent_messages,
            )
            ctx.state.evaluate_agent_messages += result.all_messages()
            
            span.set_attribute('evaluation_correct', result.output.correct)
            span.set_attribute('evaluation_comment', result.output.comment)
            
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
                return Reprimand(comment=result.output.comment)


class Reprimand(BaseNode[QuestionState]):
    """Node that handles incorrect answers and provides feedback."""
    
    comment: str = Field(
        min_length=1,
        description="Feedback comment for the incorrect answer"
    )
    
    @field_validator('comment')
    @classmethod
    def validate_comment_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Comment cannot be empty or whitespace only")
        return v.strip()

    async def run(self, ctx: GraphRunContext[QuestionState]) -> Ask:
        with logfire.span('reprimand_node.provide_feedback') as span:
            span.set_attribute('feedback_comment', self.comment)
            
            print(f'Comment: {self.comment}')
            logfire.info('Providing feedback for incorrect answer', comment=self.comment)
            
            ctx.state.question = None
            return Ask()


question_graph = Graph(
    nodes=(Ask, Answer, Evaluate, Reprimand), state_type=QuestionState
)


async def initialize_graphiti_state(
    state: Optional[QuestionState] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    enable_graphiti: bool = True
) -> QuestionState:
    """Initialize QuestionState with GraphitiClient.
    
    Args:
        state: Existing state to enhance (if None, creates new)
        user_id: User ID for the session
        session_id: Session ID (generated if not provided)
        enable_graphiti: Whether to enable Graphiti integration
        
    Returns:
        Initialized QuestionState
    """
    import uuid
    
    if state is None:
        state = QuestionState()
    
    # Set session ID
    if session_id is None:
        session_id = str(uuid.uuid4())
    state.session_id = session_id
    
    # Initialize Graphiti client if enabled
    if enable_graphiti:
        try:
            state.graphiti_client = GraphitiClient()
            await state.graphiti_client.connect()
            logfire.info(f"Initialized Graphiti client for session {session_id}")
            
            # Create or get user entity
            if user_id is None:
                user_id = f"user_{uuid.uuid4().hex[:8]}"
            
            state.current_user = UserEntity(
                id=user_id,
                session_id=session_id,
                total_questions=0,
                correct_answers=0,
                average_response_time=0.0
            )
            
        except Exception as e:
            logfire.error(f"Failed to initialize Graphiti: {e}")
            # Continue without Graphiti if it fails
            state.graphiti_client = None
            state.current_user = None
    
    return state


async def run_as_continuous():
    with logfire.span('continuous_mode.session') as span:
        logfire.info('Starting continuous mode session')
        
        # Initialize state with Graphiti
        state = await initialize_graphiti_state(enable_graphiti=True)
        
        try:
            node = Ask()
            end = await question_graph.run(node, state=state)
            
            span.set_attribute('session_result', end.output)
            span.set_attribute('session_id', state.session_id)
            logfire.info('Continuous mode session completed', result=end.output)
            
            print('END:', end.output)
            
            # Get session stats if Graphiti is enabled
            if state.graphiti_client:
                stats = state.graphiti_client.get_session_stats()
                print(f"\nSession Stats: {stats}")
                
        finally:
            # Clean up Graphiti connection
            if state.graphiti_client:
                await state.graphiti_client.disconnect()


async def run_as_cli(answer: str | None):
    with logfire.span('cli_mode.session') as span:
        persistence = FileStatePersistence(Path('question_graph.json'))
        persistence.set_graph_types(question_graph)

        if snapshot := await persistence.load_next():
            state = snapshot.state
            span.set_attribute('session_type', 'resumed')
            logfire.info('Resuming CLI session from saved state')
            
            # Re-initialize Graphiti client for resumed session
            state = await initialize_graphiti_state(
                state=state,
                session_id=state.session_id,
                enable_graphiti=True
            )
            
            assert answer is not None, (
                'answer required, usage "uv run -m pydantic_ai_examples.question_graph cli <answer>"'
            )
            node = Evaluate(answer)
        else:
            # Initialize new state with Graphiti
            state = await initialize_graphiti_state(enable_graphiti=True)
            node = Ask()
            span.set_attribute('session_type', 'new')
            logfire.info('Starting new CLI session')
        # debug(state, node)

        try:
            async with question_graph.iter(node, state=state, persistence=persistence) as run:
                while True:
                    node = await run.next()
                    if isinstance(node, End):
                        span.set_attribute('session_result', node.data)
                        span.set_attribute('session_id', state.session_id)
                        logfire.info('CLI session completed', result=node.data)
                        
                        print('END:', node.data)
                        history = await persistence.load_all()
                        print('history:', '\n'.join(str(e.node) for e in history), sep='\n')
                        
                        # Show session stats if Graphiti is enabled
                        if state.graphiti_client:
                            stats = state.graphiti_client.get_session_stats()
                            print(f"\nSession Stats: {stats}")
                        
                        print('Finished!')
                        break
                    elif isinstance(node, Answer):
                        logfire.info('Waiting for user input', question=node.question)
                        print(node.question)
                        break
                    # otherwise just continue
                    
        finally:
            # Clean up Graphiti connection
            if state.graphiti_client:
                await state.graphiti_client.disconnect()


if __name__ == '__main__':
    import asyncio
    import sys

    try:
        sub_command = sys.argv[1]
        assert sub_command in ('continuous', 'cli', 'mermaid')
    except (IndexError, AssertionError):
        print(
            'Usage:\n'
            '  uv run -m pydantic_ai_examples.question_graph mermaid\n'
            'or:\n'
            '  uv run -m pydantic_ai_examples.question_graph continuous\n'
            'or:\n'
            '  uv run -m pydantic_ai_examples.question_graph cli [answer]',
            file=sys.stderr,
        )
        sys.exit(1)

    if sub_command == 'mermaid':
        print(question_graph.mermaid_code(start_node=Ask))
    elif sub_command == 'continuous':
        asyncio.run(run_as_continuous())
    else:
        a = sys.argv[2] if len(sys.argv) > 2 else None
        asyncio.run(run_as_cli(a))
