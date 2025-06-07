"""
Example demonstrating memory retrieval for intelligent question generation.

This example shows how the enhanced question agent uses memory retrieval
to generate personalized questions based on user history and performance.
"""

import asyncio
import logging
from datetime import datetime
from typing import Optional

from graphiti_config import RuntimeConfig, get_config
from graphiti_connection import create_connection_manager
from graphiti_client import GraphitiClient
from graphiti_init import GraphitiInitializer
from memory_retrieval import MemoryRetrieval, QuestionSelector
from enhanced_question_agent import (
    EnhancedQuestionAgent,
    QuestionGenerationContext,
    MemoryAwareQuestionNode,
)
from memory_integration import (
    MemoryEnhancedAsk,
    MemoryContextBuilder,
    MemoryStateEnhancer,
    create_memory_enhanced_graph,
)
from graphiti_memory import MemoryStorage, QAPair, MemoryAnalytics
from graphiti_entities import DifficultyLevel, UserEntity
from question_graph import QuestionState
from pydantic_graph import GraphRunContext


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def setup_sample_data(client: GraphitiClient) -> None:
    """Set up sample Q&A data for demonstration."""
    logger.info("Setting up sample data...")
    
    storage = MemoryStorage(client=client)
    
    # Create sample Q&A pairs
    sample_qa_pairs = [
        QAPair(
            question="What is 2 + 2?",
            answer="4",
            user_id="demo_user",
            session_id="session_001",
            topics=["math", "arithmetic"],
            difficulty=DifficultyLevel.EASY,
            correct=True,
            response_time=2.5
        ),
        QAPair(
            question="What is the capital of France?",
            answer="Paris",
            user_id="demo_user",
            session_id="session_001",
            topics=["geography", "europe"],
            difficulty=DifficultyLevel.EASY,
            correct=True,
            response_time=3.0
        ),
        QAPair(
            question="Solve for x: 2x + 5 = 13",
            answer="x = 4",
            user_id="demo_user",
            session_id="session_001",
            topics=["math", "algebra"],
            difficulty=DifficultyLevel.MEDIUM,
            correct=True,
            response_time=15.0
        ),
        QAPair(
            question="What is the derivative of x^2?",
            answer="2x",
            user_id="demo_user",
            session_id="session_002",
            topics=["math", "calculus"],
            difficulty=DifficultyLevel.HARD,
            correct=False,
            response_time=20.0,
            evaluation_comment="Close, but remember to include dx"
        ),
        QAPair(
            question="Name the process by which plants make food",
            answer="Photosynthesis",
            user_id="demo_user",
            session_id="session_002",
            topics=["science", "biology"],
            difficulty=DifficultyLevel.MEDIUM,
            correct=True,
            response_time=5.0
        ),
        QAPair(
            question="What is the integral of 1/x?",
            answer="ln(x)",
            user_id="demo_user",
            session_id="session_002",
            topics=["math", "calculus"],
            difficulty=DifficultyLevel.HARD,
            correct=False,
            response_time=25.0,
            evaluation_comment="Don't forget the constant of integration: ln(x) + C"
        ),
    ]
    
    # Store the Q&A pairs
    successful, failed = await storage.store_batch_qa_pairs(sample_qa_pairs)
    logger.info(f"Stored {successful} Q&A pairs, {failed} failed")


async def demonstrate_memory_retrieval(client: GraphitiClient) -> None:
    """Demonstrate memory retrieval capabilities."""
    logger.info("\n=== Memory Retrieval Demonstration ===")
    
    retrieval = MemoryRetrieval(client=client)
    user_id = "demo_user"
    
    # 1. Get user performance
    logger.info("\n1. Getting user performance...")
    performance = await retrieval.get_user_performance(user_id)
    logger.info(f"User performance:")
    logger.info(f"  - Total questions: {performance['total_questions']}")
    logger.info(f"  - Accuracy: {performance['accuracy']:.1%}")
    logger.info(f"  - Recommended difficulty: {performance['recommended_difficulty']}")
    
    # 2. Get weak topics
    logger.info("\n2. Identifying weak topics...")
    weak_topics = await retrieval.get_weak_topics(user_id, threshold=0.6)
    if weak_topics:
        logger.info("Weak topics:")
        for topic, accuracy in weak_topics:
            logger.info(f"  - {topic}: {accuracy:.1%} accuracy")
    else:
        logger.info("No weak topics identified")
    
    # 3. Get asked questions
    logger.info("\n3. Retrieving previously asked questions...")
    asked_questions = await retrieval.get_asked_questions(user_id, limit=5)
    logger.info(f"Recent questions ({len(asked_questions)}):")
    for q in asked_questions[:3]:
        logger.info(f"  - {q.content} (Difficulty: {q.difficulty.value})")
    
    # 4. Get recommendations
    logger.info("\n4. Getting personalized recommendations...")
    recommendations = await retrieval.get_recommended_questions(user_id, count=3)
    if recommendations:
        logger.info("Recommended questions:")
        for i, q in enumerate(recommendations, 1):
            logger.info(f"  {i}. {q.content}")
            logger.info(f"     Topics: {', '.join(q.topics)}")
            logger.info(f"     Difficulty: {q.difficulty.value}")
    else:
        logger.info("No recommendations available")


async def demonstrate_enhanced_agent(client: GraphitiClient) -> None:
    """Demonstrate enhanced question agent."""
    logger.info("\n=== Enhanced Question Agent Demonstration ===")
    
    agent = EnhancedQuestionAgent(graphiti_client=client)
    
    # Create context for question generation
    context = QuestionGenerationContext(
        user_id="demo_user",
        session_id="session_003",
        prefer_weak_topics=True
    )
    
    logger.info("\nGenerating personalized question...")
    question = await agent.generate_question(context)
    logger.info(f"Generated question: {question}")
    
    # Show the context that was used
    retrieval = MemoryRetrieval(client=client)
    performance = await retrieval.get_user_performance("demo_user")
    weak_topics = await retrieval.get_weak_topics("demo_user")
    
    logger.info("\nGeneration context:")
    logger.info(f"  - User accuracy: {performance['accuracy']:.1%}")
    logger.info(f"  - Difficulty level: {performance['recommended_difficulty']}")
    if weak_topics:
        logger.info(f"  - Focus topics: {', '.join([t for t, _ in weak_topics[:2]])}")


async def demonstrate_memory_analytics(client: GraphitiClient) -> None:
    """Demonstrate memory analytics capabilities."""
    logger.info("\n=== Memory Analytics Demonstration ===")
    
    analytics = MemoryAnalytics(client=client)
    
    # Get user stats
    logger.info("\nUser Statistics:")
    stats = await analytics.get_user_stats("demo_user")
    if stats:
        logger.info(f"  - Total questions: {stats.get('total_questions', 0)}")
        logger.info(f"  - Correct answers: {stats.get('correct_answers', 0)}")
        logger.info(f"  - Overall accuracy: {stats.get('accuracy', 0):.1%}")
        logger.info(f"  - Avg response time: {stats.get('avg_response_time', 0):.1f}s")
    
    # Get topic performance
    logger.info("\nTopic Performance:")
    for topic in ["math", "science", "geography"]:
        perf = await analytics.get_topic_performance("demo_user", topic)
        if perf and perf.get('total_questions', 0) > 0:
            logger.info(f"  {topic}:")
            logger.info(f"    - Questions: {perf['total_questions']}")
            logger.info(f"    - Accuracy: {perf['accuracy']:.1%}")
            logger.info(f"    - Avg time: {perf.get('avg_response_time', 0):.1f}s")
    
    # Get session summary
    logger.info("\nSession Summaries:")
    for session_id in ["session_001", "session_002"]:
        summary = await analytics.get_session_summary(session_id)
        if summary:
            logger.info(f"  {session_id}:")
            logger.info(f"    - Questions: {summary['questions_asked']}")
            logger.info(f"    - Accuracy: {summary['accuracy']:.1%}")
            logger.info(f"    - Topics: {', '.join(summary.get('topics_covered', []))}")


async def demonstrate_memory_enhanced_graph(client: GraphitiClient) -> None:
    """Demonstrate memory-enhanced graph execution."""
    logger.info("\n=== Memory-Enhanced Graph Demonstration ===")
    
    # Create enhanced state
    state = QuestionState()
    state = MemoryStateEnhancer.enhance_state(
        state,
        graphiti_client=client,
        user_id="demo_user",
        session_id="session_demo"
    )
    
    # Create memory-enhanced graph
    graph = await create_memory_enhanced_graph()
    
    # Create context
    ctx = Mock(spec=GraphRunContext)
    ctx.state = state
    
    # Run the memory-enhanced ask node
    logger.info("\nRunning memory-enhanced Ask node...")
    node = MemoryEnhancedAsk()
    
    # Create a simple mock context
    class MockContext:
        def __init__(self, state):
            self.state = state
    
    ctx = MockContext(state)
    
    try:
        result = await node.run(ctx)
        logger.info(f"Generated question: {result.question}")
        logger.info(f"Question stored in state: {state.question}")
    except Exception as e:
        logger.error(f"Error running node: {e}")


async def main():
    """Main demonstration function."""
    logger.info("Starting Memory Retrieval Demonstration")
    
    # Initialize configuration
    config = RuntimeConfig(
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="password123"
    )
    
    # Create connection manager
    connection_manager = create_connection_manager(config)
    
    # Create Graphiti client
    logger.info("Creating Graphiti client...")
    client = GraphitiClient(connection_manager=connection_manager)
    
    # Initialize database
    logger.info("Initializing database...")
    initializer = GraphitiInitializer(connection_manager=connection_manager)
    if await initializer.initialize():
        logger.info("Database initialized successfully")
    
    try:
        # Set up sample data
        await setup_sample_data(client)
        
        # Run demonstrations
        await demonstrate_memory_retrieval(client)
        await demonstrate_enhanced_agent(client)
        await demonstrate_memory_analytics(client)
        # await demonstrate_memory_enhanced_graph(client)  # Commented due to mock complexity
        
        logger.info("\n=== Demonstration Complete ===")
        
    except Exception as e:
        logger.error(f"Error during demonstration: {e}")
        raise
    
    finally:
        # Clean up
        await connection_manager.close()


if __name__ == "__main__":
    # Note: This requires a running Neo4j instance
    # You can start one with: make docker-neo4j
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Demonstration interrupted by user")
    except Exception as e:
        logger.error(f"Demonstration failed: {e}")
        
        # If Neo4j is not running, provide helpful message
        if "Failed to establish connection" in str(e):
            logger.info("\nTo run this example, you need a Neo4j instance running.")
            logger.info("Start one with: make docker-neo4j")
            logger.info("Default connection: bolt://localhost:7687")
            logger.info("Default credentials: neo4j/password123")