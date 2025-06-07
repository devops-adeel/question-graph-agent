"""
Enhanced question agent that uses memory retrieval for intelligent question generation.

This module provides an enhanced version of the question-asking agent that
leverages the Graphiti knowledge graph to generate context-aware, personalized
questions based on user history and performance.
"""

import logging
from typing import Optional, List, Dict, Any
from datetime import datetime

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import ModelMessage

from graphiti_client import GraphitiClient
from memory_retrieval import MemoryRetrieval, QuestionSelector
from graphiti_entities import DifficultyLevel, UserEntity


logger = logging.getLogger(__name__)


class QuestionGenerationContext(BaseModel):
    """Context for question generation."""
    user_id: str = Field(description="User ID for personalization")
    session_id: Optional[str] = Field(default=None, description="Current session ID")
    max_attempts: int = Field(default=3, description="Maximum generation attempts")
    avoid_recent: int = Field(default=10, description="Number of recent questions to avoid")
    prefer_weak_topics: bool = Field(default=True, description="Prefer questions from weak topics")
    difficulty_override: Optional[DifficultyLevel] = Field(default=None, description="Override difficulty")


class EnhancedQuestionAgent:
    """Enhanced agent that generates questions using memory retrieval."""
    
    def __init__(self,
                 model: str = "openai:gpt-4o",
                 graphiti_client: Optional[GraphitiClient] = None):
        """Initialize enhanced question agent.
        
        Args:
            model: Model to use for generation
            graphiti_client: Graphiti client for memory access
        """
        self.model = model
        self.graphiti_client = graphiti_client
        self.retrieval = MemoryRetrieval(client=graphiti_client)
        self.selector = QuestionSelector(self.retrieval)
        
        # Create the underlying pydantic_ai agent
        self.agent = Agent(
            model,
            system_prompt=self._build_system_prompt(),
            retries=2
        )
    
    def _build_system_prompt(self) -> str:
        """Build the system prompt for the agent."""
        return """You are an intelligent question-asking agent that generates educational questions.

Your goal is to create questions that:
1. Are clear, concise, and have a single correct answer
2. Match the user's skill level and learning needs
3. Build on previous knowledge while introducing new concepts
4. Avoid repetition of recently asked questions
5. Focus on areas where the user needs improvement

You will receive context about:
- Previously asked questions to avoid
- User's performance and recommended difficulty
- Weak topics that need practice
- Specific requirements for the question

Generate questions that are educational, engaging, and appropriate for the user's level."""
    
    async def generate_question(self,
                                context: QuestionGenerationContext,
                                message_history: Optional[List[ModelMessage]] = None) -> str:
        """Generate a question using memory-informed context.
        
        Args:
            context: Question generation context
            message_history: Optional message history
            
        Returns:
            Generated question
        """
        try:
            # First, try to select from existing questions
            if self.graphiti_client:
                selected_question = await self.selector.select_next_question(
                    user_id=context.user_id,
                    session_id=context.session_id
                )
                
                if selected_question:
                    logger.info(f"Selected existing question for user {context.user_id}")
                    return selected_question
            
            # If no suitable existing question, generate new one
            generation_context = await self._build_generation_context(context)
            
            prompt = self._build_generation_prompt(generation_context)
            
            result = await self.agent.run(
                prompt,
                message_history=message_history or []
            )
            
            generated_question = result.data
            logger.info(f"Generated new question for user {context.user_id}")
            
            return generated_question
            
        except Exception as e:
            logger.error(f"Failed to generate question: {e}")
            # Fallback to simple generation
            return await self._fallback_generation(message_history)
    
    async def _build_generation_context(self, 
                                        context: QuestionGenerationContext) -> Dict[str, Any]:
        """Build context for question generation.
        
        Args:
            context: Question generation context
            
        Returns:
            Dictionary with generation context
        """
        generation_context = {
            "user_id": context.user_id,
            "session_id": context.session_id
        }
        
        if not self.graphiti_client:
            return generation_context
        
        # Get user performance
        performance = await self.retrieval.get_user_performance(context.user_id)
        generation_context["user_performance"] = performance
        generation_context["recommended_difficulty"] = context.difficulty_override or performance["recommended_difficulty"]
        
        # Get recent questions to avoid
        recent_questions = await self.retrieval.get_asked_questions(
            context.user_id,
            context.session_id,
            limit=context.avoid_recent
        )
        generation_context["avoid_questions"] = [q.content for q in recent_questions]
        
        # Get weak topics if preferred
        if context.prefer_weak_topics:
            weak_topics = await self.retrieval.get_weak_topics(context.user_id)
            generation_context["weak_topics"] = [topic for topic, _ in weak_topics[:5]]
        
        return generation_context
    
    def _build_generation_prompt(self, context: Dict[str, Any]) -> str:
        """Build the prompt for question generation.
        
        Args:
            context: Generation context
            
        Returns:
            Formatted prompt
        """
        prompt_parts = ["Generate an educational question with the following requirements:"]
        
        # Add difficulty requirement
        difficulty = context.get("recommended_difficulty", DifficultyLevel.MEDIUM)
        prompt_parts.append(f"\nDifficulty: {difficulty.value}")
        
        # Add topics to focus on
        if "weak_topics" in context and context["weak_topics"]:
            topics_str = ", ".join(context["weak_topics"][:3])
            prompt_parts.append(f"\nFocus on these topics (user needs practice): {topics_str}")
        
        # Add questions to avoid
        if "avoid_questions" in context and context["avoid_questions"]:
            prompt_parts.append("\n\nAvoid questions similar to these recently asked ones:")
            for i, q in enumerate(context["avoid_questions"][:5], 1):
                prompt_parts.append(f"\n{i}. {q}")
        
        # Add performance context
        if "user_performance" in context:
            perf = context["user_performance"]
            accuracy = perf.get("accuracy", 0) * 100
            prompt_parts.append(f"\n\nUser's current accuracy: {accuracy:.1f}%")
            
            if accuracy < 50:
                prompt_parts.append("The user is struggling, so keep questions clear and straightforward.")
            elif accuracy > 80:
                prompt_parts.append("The user is performing well, so you can include more challenging concepts.")
        
        prompt_parts.append("\n\nGenerate a single question that helps the user learn and improve.")
        
        return "\n".join(prompt_parts)
    
    async def _fallback_generation(self, 
                                   message_history: Optional[List[ModelMessage]] = None) -> str:
        """Fallback question generation without memory context.
        
        Args:
            message_history: Optional message history
            
        Returns:
            Generated question
        """
        try:
            result = await self.agent.run(
                "Generate a simple educational question with a single correct answer. "
                "Make it clear and appropriate for general learning.",
                message_history=message_history or []
            )
            return result.data
        except Exception as e:
            logger.error(f"Fallback generation failed: {e}")
            # Ultimate fallback
            return "What is 2 + 2?"


class MemoryAwareQuestionNode:
    """Graph node that uses memory-aware question generation."""
    
    def __init__(self, graphiti_client: Optional[GraphitiClient] = None):
        """Initialize memory-aware question node.
        
        Args:
            graphiti_client: Graphiti client for memory access
        """
        self.agent = EnhancedQuestionAgent(graphiti_client=graphiti_client)
    
    async def generate_question(self,
                                user_id: str,
                                session_id: Optional[str] = None,
                                message_history: Optional[List[ModelMessage]] = None) -> str:
        """Generate a question for the user.
        
        Args:
            user_id: User ID
            session_id: Optional session ID
            message_history: Optional message history
            
        Returns:
            Generated question
        """
        context = QuestionGenerationContext(
            user_id=user_id,
            session_id=session_id
        )
        
        return await self.agent.generate_question(context, message_history)