"""
Entity extraction module for converting Q&A text into structured entities.

This module provides extraction capabilities for identifying topics, entities,
and relationships from natural language questions and answers.
"""

from __future__ import annotations

import asyncio
import re
from collections import Counter
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple

import logfire
from pydantic import BaseModel, Field

# Import entity models
from graphiti_entities import (
    AnswerEntity,
    AnswerStatus,
    DifficultyLevel,
    EntityFactory,
    QuestionEntity,
    TopicEntity,
    UserEntity,
)


# Sub-task 3.2: Difficulty estimation metrics
class ComplexityMetrics(BaseModel):
    """Metrics for estimating question complexity."""
    word_count: int = Field(ge=0)
    unique_words: int = Field(ge=0)
    avg_word_length: float = Field(ge=0)
    sentence_count: int = Field(ge=0)
    technical_term_count: int = Field(ge=0)
    math_symbol_count: int = Field(ge=0)
    nested_clause_count: int = Field(ge=0)


class EntityExtractor:
    """Main entity extractor for questions and answers."""
    
    def __init__(self):
        """Initialize the entity extractor."""
        self.factory = EntityFactory()
        self._init_patterns()
        self._init_technical_terms()
    
    def _init_patterns(self):
        """Initialize regex patterns for extraction."""
        self.patterns = {
            'math_symbols': re.compile(r'[+\-*/=<>≤≥∑∏∫∂√π±×÷≠≈∞]'),
            'numbers': re.compile(r'\b\d+\.?\d*\b'),
            'technical_terms': re.compile(r'\b(algorithm|function|variable|equation|formula|theorem|proof|matrix|vector|derivative|integral|limit)\b', re.IGNORECASE),
            'nested_clauses': re.compile(r'(which|that|who|whom|whose|where|when|while|if|unless|because|although|though)'),
            'question_words': re.compile(r'^(what|who|where|when|why|how|which|whose)\b', re.IGNORECASE),
        }
    
    def _init_technical_terms(self):
        """Initialize technical term dictionary."""
        self.technical_terms = {
            'mathematics': {'algebra', 'calculus', 'geometry', 'trigonometry', 'statistics', 'probability'},
            'programming': {'python', 'java', 'javascript', 'function', 'class', 'variable', 'loop', 'algorithm'},
            'science': {'physics', 'chemistry', 'biology', 'hypothesis', 'experiment', 'theory', 'law'},
            'history': {'war', 'revolution', 'empire', 'civilization', 'dynasty', 'era', 'period'},
            'geography': {'continent', 'country', 'capital', 'ocean', 'mountain', 'river', 'climate'},
        }
    
    async def extract_from_text(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """Extract entities from text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of extracted entities by type
        """
        # Extract topics
        topics = self._extract_topics(text)
        
        # Extract other entities (simplified)
        entities = {
            'topics': [{'name': topic} for topic in topics],
            'concepts': [],
            'references': []
        }
        
        return entities
    
    def _extract_topics(self, text: str) -> List[str]:
        """Extract topics from text."""
        topics = []
        text_lower = text.lower()
        
        # Check for technical terms in each category
        for category, terms in self.technical_terms.items():
            for term in terms:
                if term in text_lower:
                    topics.append(category)
                    break
        
        return list(set(topics))  # Remove duplicates
    
    def estimate_difficulty(self, text: str) -> DifficultyLevel:
        """Estimate difficulty level of text.
        
        Args:
            text: Input text
            
        Returns:
            Estimated difficulty level
        """
        metrics = self._calculate_complexity_metrics(text)
        
        # Simple heuristic for difficulty
        score = 0
        
        # Word count factor
        if metrics.word_count > 20:
            score += 1
        if metrics.word_count > 40:
            score += 1
            
        # Technical terms factor
        if metrics.technical_term_count > 2:
            score += 1
        if metrics.technical_term_count > 5:
            score += 1
            
        # Math symbols factor
        if metrics.math_symbol_count > 0:
            score += 1
        if metrics.math_symbol_count > 3:
            score += 1
            
        # Average word length factor
        if metrics.avg_word_length > 5:
            score += 1
        if metrics.avg_word_length > 7:
            score += 1
            
        # Map score to difficulty
        if score <= 2:
            return DifficultyLevel.EASY
        elif score <= 5:
            return DifficultyLevel.MEDIUM
        else:
            return DifficultyLevel.HARD
    
    def _calculate_complexity_metrics(self, text: str) -> ComplexityMetrics:
        """Calculate complexity metrics for text."""
        words = text.split()
        word_count = len(words)
        unique_words = len(set(words))
        
        # Calculate average word length
        total_length = sum(len(word.strip('.,!?;:')) for word in words)
        avg_word_length = total_length / word_count if word_count > 0 else 0
        
        # Count sentences (simple approximation)
        sentence_count = len(re.split(r'[.!?]+', text.strip())) - 1
        if sentence_count <= 0:
            sentence_count = 1
        
        # Count technical terms
        technical_term_count = len(self.patterns['technical_terms'].findall(text))
        
        # Count math symbols
        math_symbol_count = len(self.patterns['math_symbols'].findall(text))
        
        # Count nested clauses
        nested_clause_count = len(self.patterns['nested_clauses'].findall(text))
        
        return ComplexityMetrics(
            word_count=word_count,
            unique_words=unique_words,
            avg_word_length=avg_word_length,
            sentence_count=sentence_count,
            technical_term_count=technical_term_count,
            math_symbol_count=math_symbol_count,
            nested_clause_count=nested_clause_count
        )


# Global instance for convenience
default_extractor = EntityExtractor()