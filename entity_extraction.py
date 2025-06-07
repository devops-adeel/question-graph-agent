"""
Entity extraction module for converting Q&A text into structured entities.

This module provides extraction capabilities for identifying topics, entities,
and relationships from natural language questions and answers.
"""

from __future__ import annotations

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
    """Metrics for measuring question complexity."""
    
    word_count: int = Field(ge=0)
    avg_word_length: float = Field(ge=0.0)
    sentence_count: int = Field(ge=1)
    avg_sentence_length: float = Field(ge=0.0)
    
    # Linguistic complexity
    complex_word_count: int = Field(ge=0)  # Words > 6 characters
    technical_term_count: int = Field(ge=0)
    subordinate_clause_count: int = Field(ge=0)
    
    # Question type indicators
    has_negation: bool = False
    has_comparison: bool = False
    has_multiple_parts: bool = False
    requires_calculation: bool = False
    requires_reasoning: bool = False
    
    # Cognitive load
    concept_count: int = Field(ge=0)
    relationship_count: int = Field(ge=0)
    abstraction_level: float = Field(ge=0.0, le=1.0)
    
    # Overall scores
    linguistic_complexity: float = Field(ge=0.0, le=1.0)
    cognitive_complexity: float = Field(ge=0.0, le=1.0)
    overall_complexity: float = Field(ge=0.0, le=1.0)


# Sub-task 3.1: Basic EntityExtractor class
class TopicKeywords(BaseModel):
    """Topic keyword mapping for extraction."""
    
    topic_name: str
    primary_keywords: List[str] = Field(default_factory=list)
    secondary_keywords: List[str] = Field(default_factory=list)
    regex_patterns: List[str] = Field(default_factory=list)
    weight: float = Field(default=1.0, ge=0.0, le=2.0)


class ExtractedTopic(BaseModel):
    """Result of topic extraction."""
    
    topic_name: str
    confidence: float = Field(ge=0.0, le=1.0)
    matched_keywords: List[str] = Field(default_factory=list)
    match_type: str = Field(default="keyword")  # keyword, pattern, inference
    position_in_text: Optional[int] = None


class EntityExtractor:
    """
    Extract entities from Q&A text using keyword and pattern matching.
    
    This is a basic implementation that can be enhanced with NLP models.
    """
    
    def __init__(self, custom_topics: Optional[List[TopicKeywords]] = None):
        """
        Initialize the extractor with topic definitions.
        
        Args:
            custom_topics: Optional custom topic definitions
        """
        self.topics = self._initialize_default_topics()
        if custom_topics:
            self._add_custom_topics(custom_topics)
        
        # Compile regex patterns for efficiency
        self._compiled_patterns: Dict[str, List[re.Pattern]] = {}
        self._compile_patterns()
        
        # Cache for performance
        self._extraction_cache: Dict[str, List[ExtractedTopic]] = {}
    
    def _initialize_default_topics(self) -> Dict[str, TopicKeywords]:
        """Initialize default topic keyword mappings."""
        default_topics = [
            TopicKeywords(
                topic_name="mathematics",
                primary_keywords=["math", "calculate", "equation", "formula", "solve"],
                secondary_keywords=["number", "arithmetic", "algebra", "geometry", "calculus"],
                regex_patterns=[r"\d+\s*[\+\-\*\/]\s*\d+", r"x\s*=\s*\d+"],
                weight=1.0
            ),
            TopicKeywords(
                topic_name="science",
                primary_keywords=["scientific", "experiment", "hypothesis", "theory", "research"],
                secondary_keywords=["study", "observe", "test", "prove", "evidence"],
                regex_patterns=[r"H[0-9]+O[0-9]*", r"[A-Z][a-z]?\d*"],  # Chemical formulas
                weight=1.0
            ),
            TopicKeywords(
                topic_name="history",
                primary_keywords=["historical", "history", "ancient", "war", "civilization"],
                secondary_keywords=["century", "era", "period", "dynasty", "empire"],
                regex_patterns=[r"\b\d{4}\s*(BCE?|CE|AD|BC)\b", r"\b\d{1,2}th\s+century\b"],
                weight=1.0
            ),
            TopicKeywords(
                topic_name="geography",
                primary_keywords=["country", "continent", "capital", "city", "location"],
                secondary_keywords=["map", "region", "territory", "border", "ocean"],
                regex_patterns=[r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b"],  # Proper nouns
                weight=1.0
            ),
            TopicKeywords(
                topic_name="literature",
                primary_keywords=["author", "novel", "poem", "story", "literary"],
                secondary_keywords=["write", "book", "character", "plot", "theme"],
                regex_patterns=[r'"[^"]+?"', r"'[^']+?'"],  # Quoted text
                weight=0.9
            ),
            TopicKeywords(
                topic_name="technology",
                primary_keywords=["computer", "software", "algorithm", "programming", "digital"],
                secondary_keywords=["code", "system", "network", "data", "internet"],
                regex_patterns=[r"\b[A-Z]+\b(?:\s+[A-Z]+)*", r"\.com\b", r"@\w+"],
                weight=1.1
            ),
            TopicKeywords(
                topic_name="biology",
                primary_keywords=["cell", "organism", "species", "evolution", "genetic"],
                secondary_keywords=["life", "living", "plant", "animal", "dna"],
                regex_patterns=[r"\b[A-Z][a-z]+\s+[a-z]+\b"],  # Scientific names
                weight=1.0
            ),
            TopicKeywords(
                topic_name="physics",
                primary_keywords=["force", "energy", "motion", "quantum", "particle"],
                secondary_keywords=["speed", "velocity", "mass", "gravity", "wave"],
                regex_patterns=[r"E\s*=\s*mc", r"\d+\s*m/s"],
                weight=1.0
            ),
            TopicKeywords(
                topic_name="chemistry",
                primary_keywords=["element", "compound", "reaction", "molecule", "atom"],
                secondary_keywords=["chemical", "acid", "base", "ion", "bond"],
                regex_patterns=[r"[A-Z][a-z]?\d*", r"pH\s*\d+"],
                weight=1.0
            ),
        ]
        
        return {topic.topic_name: topic for topic in default_topics}
    
    def _add_custom_topics(self, custom_topics: List[TopicKeywords]):
        """Add custom topic definitions."""
        for topic in custom_topics:
            self.topics[topic.topic_name] = topic
    
    def _compile_patterns(self):
        """Compile regex patterns for all topics."""
        for topic_name, topic in self.topics.items():
            if topic.regex_patterns:
                self._compiled_patterns[topic_name] = [
                    re.compile(pattern, re.IGNORECASE)
                    for pattern in topic.regex_patterns
                ]
    
    def extract_topics(
        self,
        text: str,
        min_confidence: float = 0.3,
        max_topics: int = 5
    ) -> List[ExtractedTopic]:
        """
        Extract topics from text using keyword and pattern matching.
        
        Args:
            text: Text to extract topics from
            min_confidence: Minimum confidence threshold
            max_topics: Maximum number of topics to return
            
        Returns:
            List of extracted topics sorted by confidence
        """
        # Check cache
        cache_key = f"{text[:100]}_{min_confidence}_{max_topics}"
        if cache_key in self._extraction_cache:
            return self._extraction_cache[cache_key]
        
        with logfire.span("entity_extractor.extract_topics") as span:
            span.set_attribute("text_length", len(text))
            
            # Preprocess text
            text_lower = text.lower()
            words = re.findall(r'\b\w+\b', text_lower)
            word_set = set(words)
            
            # Extract topics
            extracted_topics = []
            
            for topic_name, topic_def in self.topics.items():
                confidence = 0.0
                matched_keywords = []
                match_type = "none"
                
                # Check primary keywords
                primary_matches = []
                for keyword in topic_def.primary_keywords:
                    if keyword in text_lower:
                        primary_matches.append(keyword)
                        # Check position for better scoring
                        position = text_lower.find(keyword)
                        position_score = 1.0 - (position / len(text_lower))
                        confidence += 0.3 * position_score
                
                # Check secondary keywords
                secondary_matches = []
                for keyword in topic_def.secondary_keywords:
                    if keyword in word_set:
                        secondary_matches.append(keyword)
                        confidence += 0.1
                
                # Check regex patterns
                pattern_matches = []
                if topic_name in self._compiled_patterns:
                    for pattern in self._compiled_patterns[topic_name]:
                        if pattern.search(text):
                            pattern_matches.append(pattern.pattern)
                            confidence += 0.2
                
                # Determine match type and final confidence
                if primary_matches:
                    match_type = "keyword"
                    matched_keywords = primary_matches + secondary_matches
                elif pattern_matches:
                    match_type = "pattern"
                    matched_keywords = pattern_matches
                elif secondary_matches:
                    match_type = "inference"
                    matched_keywords = secondary_matches
                    confidence *= 0.7  # Lower confidence for inference
                
                # Apply topic weight
                confidence *= topic_def.weight
                
                # Cap confidence at 1.0
                confidence = min(1.0, confidence)
                
                if confidence >= min_confidence:
                    extracted_topics.append(ExtractedTopic(
                        topic_name=topic_name,
                        confidence=confidence,
                        matched_keywords=matched_keywords[:5],  # Limit keywords
                        match_type=match_type,
                        position_in_text=text_lower.find(matched_keywords[0]) if matched_keywords else None
                    ))
            
            # Sort by confidence and limit
            extracted_topics.sort(key=lambda x: x.confidence, reverse=True)
            result = extracted_topics[:max_topics]
            
            # Cache result
            self._extraction_cache[cache_key] = result
            
            span.set_attribute("topics_found", len(result))
            logfire.info(
                "Topics extracted",
                text_preview=text[:50],
                topics=[t.topic_name for t in result]
            )
            
            return result
    
    def extract_entities_from_question(
        self,
        question_text: str,
        question_id: Optional[str] = None
    ) -> QuestionEntity:
        """
        Extract a QuestionEntity from question text with difficulty estimation.
        
        Args:
            question_text: The question text
            question_id: Optional ID for the question
            
        Returns:
            Extracted QuestionEntity with topics and difficulty
        """
        # Extract topics
        extracted_topics = self.extract_topics(question_text)
        topic_names = [t.topic_name for t in extracted_topics]
        
        # Estimate difficulty
        difficulty_level, difficulty_score, metrics = self.estimate_difficulty(
            question_text, 
            extracted_topics
        )
        
        # Create question entity using factory
        question = EntityFactory.create_question_from_text(
            content=question_text,
            topics=topic_names if topic_names else ["general"],
            difficulty=difficulty_level
        )
        
        # Override ID if provided
        if question_id:
            question.id = question_id
        
        # Log extraction results
        logfire.info(
            "Question entity extracted",
            question_id=question.id,
            topics=topic_names,
            difficulty=difficulty_level.value,
            word_count=metrics.word_count,
            cognitive_complexity=metrics.cognitive_complexity
        )
        
        return question
    
    def extract_concepts_from_text(
        self,
        text: str,
        topic_filter: Optional[List[str]] = None
    ) -> List[str]:
        """
        Extract key concepts from text.
        
        Args:
            text: Text to analyze
            topic_filter: Optional list of topics to focus on
            
        Returns:
            List of extracted concepts
        """
        concepts = []
        
        # Extract noun phrases (simple approach)
        # Pattern: Adjective* Noun+
        noun_phrase_pattern = re.compile(
            r'\b(?:[A-Za-z]+\s+)*[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
        )
        
        matches = noun_phrase_pattern.findall(text)
        
        # Filter by topic keywords if specified
        if topic_filter:
            relevant_keywords = set()
            for topic_name in topic_filter:
                if topic_name in self.topics:
                    topic = self.topics[topic_name]
                    relevant_keywords.update(topic.primary_keywords)
                    relevant_keywords.update(topic.secondary_keywords)
            
            # Keep concepts that contain relevant keywords
            for match in matches:
                match_lower = match.lower()
                if any(keyword in match_lower for keyword in relevant_keywords):
                    concepts.append(match)
        else:
            concepts = matches
        
        # Deduplicate while preserving order
        seen = set()
        unique_concepts = []
        for concept in concepts:
            if concept.lower() not in seen:
                seen.add(concept.lower())
                unique_concepts.append(concept)
        
        return unique_concepts[:10]  # Limit to top 10
    
    def calculate_topic_overlap(
        self,
        topics1: List[ExtractedTopic],
        topics2: List[ExtractedTopic]
    ) -> float:
        """
        Calculate overlap between two sets of topics.
        
        Args:
            topics1: First set of topics
            topics2: Second set of topics
            
        Returns:
            Overlap score (0-1)
        """
        if not topics1 or not topics2:
            return 0.0
        
        # Create topic sets with confidence weights
        topics1_weighted = {t.topic_name: t.confidence for t in topics1}
        topics2_weighted = {t.topic_name: t.confidence for t in topics2}
        
        # Calculate weighted overlap
        common_topics = set(topics1_weighted.keys()) & set(topics2_weighted.keys())
        
        if not common_topics:
            return 0.0
        
        overlap_score = sum(
            min(topics1_weighted[topic], topics2_weighted[topic])
            for topic in common_topics
        )
        
        max_score = min(
            sum(topics1_weighted.values()),
            sum(topics2_weighted.values())
        )
        
        return overlap_score / max_score if max_score > 0 else 0.0
    
    def suggest_related_topics(
        self,
        primary_topic: str,
        extracted_topics: List[ExtractedTopic]
    ) -> List[str]:
        """
        Suggest related topics based on extraction results.
        
        Args:
            primary_topic: Main topic
            extracted_topics: Already extracted topics
            
        Returns:
            List of suggested related topics
        """
        suggestions = []
        
        # Topic relationships (simplified)
        topic_relationships = {
            "mathematics": ["physics", "technology", "science"],
            "physics": ["mathematics", "chemistry", "technology"],
            "chemistry": ["physics", "biology", "science"],
            "biology": ["chemistry", "science", "medicine"],
            "history": ["geography", "literature", "politics"],
            "geography": ["history", "geology", "culture"],
            "literature": ["history", "language", "culture"],
            "technology": ["mathematics", "physics", "engineering"],
            "science": ["mathematics", "physics", "chemistry", "biology"]
        }
        
        # Get related topics
        if primary_topic in topic_relationships:
            related = topic_relationships[primary_topic]
            
            # Filter out already extracted topics
            extracted_names = {t.topic_name for t in extracted_topics}
            suggestions = [t for t in related if t not in extracted_names]
        
        return suggestions[:3]  # Return top 3 suggestions
    
    def clear_cache(self):
        """Clear the extraction cache."""
        self._extraction_cache.clear()
        logfire.info("Entity extraction cache cleared")
    
    # Sub-task 3.2: Difficulty estimation methods
    def analyze_complexity(self, text: str) -> ComplexityMetrics:
        """
        Analyze text complexity using multiple metrics.
        
        Args:
            text: Text to analyze
            
        Returns:
            ComplexityMetrics with detailed measurements
        """
        with logfire.span("entity_extractor.analyze_complexity") as span:
            # Basic text statistics
            words = re.findall(r'\b\w+\b', text)
            word_count = len(words)
            
            # Sentence detection (simple approach)
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
            sentence_count = max(1, len(sentences))
            
            # Word complexity
            complex_words = [w for w in words if len(w) > 6]
            complex_word_count = len(complex_words)
            
            # Average calculations
            avg_word_length = sum(len(w) for w in words) / word_count if word_count > 0 else 0
            avg_sentence_length = word_count / sentence_count
            
            # Technical terms (based on topic keywords)
            technical_terms = set()
            for topic in self.topics.values():
                technical_terms.update(topic.primary_keywords)
                technical_terms.update(topic.secondary_keywords)
            
            technical_term_count = sum(
                1 for w in words 
                if w.lower() in technical_terms
            )
            
            # Question type indicators
            text_lower = text.lower()
            has_negation = any(neg in text_lower for neg in ['not', 'no', 'never', 'neither', 'none'])
            has_comparison = any(comp in text_lower for comp in ['more', 'less', 'better', 'worse', 'than'])
            has_multiple_parts = ' and ' in text_lower or ' or ' in text_lower or ';' in text
            requires_calculation = any(calc in text_lower for calc in ['calculate', 'compute', 'solve', 'find the value'])
            requires_reasoning = any(reason in text_lower for reason in ['why', 'how', 'explain', 'analyze', 'compare'])
            
            # Subordinate clauses (simple detection)
            subordinate_markers = ['because', 'although', 'while', 'when', 'if', 'since', 'unless']
            subordinate_clause_count = sum(
                1 for marker in subordinate_markers 
                if marker in text_lower
            )
            
            # Concept extraction
            concepts = self.extract_concepts_from_text(text)
            concept_count = len(concepts)
            
            # Relationship indicators
            relationship_markers = ['between', 'among', 'related to', 'connected', 'causes', 'leads to']
            relationship_count = sum(
                1 for marker in relationship_markers 
                if marker in text_lower
            )
            
            # Abstraction level (based on abstract vs concrete words)
            abstract_indicators = ['concept', 'theory', 'idea', 'principle', 'hypothesis', 'abstract']
            concrete_indicators = ['object', 'person', 'place', 'specific', 'example', 'instance']
            
            abstract_score = sum(1 for ind in abstract_indicators if ind in text_lower)
            concrete_score = sum(1 for ind in concrete_indicators if ind in text_lower)
            
            abstraction_level = abstract_score / (abstract_score + concrete_score + 1)
            
            # Calculate complexity scores
            linguistic_complexity = self._calculate_linguistic_complexity(
                word_count, avg_word_length, avg_sentence_length,
                complex_word_count, subordinate_clause_count
            )
            
            cognitive_complexity = self._calculate_cognitive_complexity(
                concept_count, relationship_count, abstraction_level,
                has_negation, has_comparison, has_multiple_parts,
                requires_calculation, requires_reasoning
            )
            
            overall_complexity = (linguistic_complexity + cognitive_complexity) / 2
            
            metrics = ComplexityMetrics(
                word_count=word_count,
                avg_word_length=avg_word_length,
                sentence_count=sentence_count,
                avg_sentence_length=avg_sentence_length,
                complex_word_count=complex_word_count,
                technical_term_count=technical_term_count,
                subordinate_clause_count=subordinate_clause_count,
                has_negation=has_negation,
                has_comparison=has_comparison,
                has_multiple_parts=has_multiple_parts,
                requires_calculation=requires_calculation,
                requires_reasoning=requires_reasoning,
                concept_count=concept_count,
                relationship_count=relationship_count,
                abstraction_level=abstraction_level,
                linguistic_complexity=linguistic_complexity,
                cognitive_complexity=cognitive_complexity,
                overall_complexity=overall_complexity
            )
            
            span.set_attribute("overall_complexity", overall_complexity)
            return metrics
    
    def _calculate_linguistic_complexity(
        self,
        word_count: int,
        avg_word_length: float,
        avg_sentence_length: float,
        complex_word_count: int,
        subordinate_clause_count: int
    ) -> float:
        """Calculate linguistic complexity score (0-1)."""
        # Normalize factors
        word_count_score = min(1.0, word_count / 50)  # 50+ words is max complexity
        word_length_score = min(1.0, (avg_word_length - 3) / 5)  # 8+ avg length is max
        sentence_length_score = min(1.0, avg_sentence_length / 25)  # 25+ words/sentence is max
        complex_word_ratio = complex_word_count / word_count if word_count > 0 else 0
        subordinate_score = min(1.0, subordinate_clause_count / 3)  # 3+ clauses is max
        
        # Weighted average
        linguistic_complexity = (
            word_count_score * 0.15 +
            word_length_score * 0.25 +
            sentence_length_score * 0.20 +
            complex_word_ratio * 0.25 +
            subordinate_score * 0.15
        )
        
        return min(1.0, linguistic_complexity)
    
    def _calculate_cognitive_complexity(
        self,
        concept_count: int,
        relationship_count: int,
        abstraction_level: float,
        has_negation: bool,
        has_comparison: bool,
        has_multiple_parts: bool,
        requires_calculation: bool,
        requires_reasoning: bool
    ) -> float:
        """Calculate cognitive complexity score (0-1)."""
        # Concept density
        concept_score = min(1.0, concept_count / 5)  # 5+ concepts is max
        relationship_score = min(1.0, relationship_count / 3)  # 3+ relationships is max
        
        # Boolean factors
        negation_score = 0.2 if has_negation else 0.0
        comparison_score = 0.2 if has_comparison else 0.0
        multipart_score = 0.3 if has_multiple_parts else 0.0
        calculation_score = 0.4 if requires_calculation else 0.0
        reasoning_score = 0.4 if requires_reasoning else 0.0
        
        # Combine scores
        cognitive_complexity = (
            concept_score * 0.2 +
            relationship_score * 0.15 +
            abstraction_level * 0.15 +
            (negation_score + comparison_score + multipart_score + 
             calculation_score + reasoning_score) * 0.5 / 1.5  # Normalize boolean scores
        )
        
        return min(1.0, cognitive_complexity)
    
    def estimate_difficulty(
        self,
        text: str,
        topics: Optional[List[ExtractedTopic]] = None
    ) -> Tuple[DifficultyLevel, float, ComplexityMetrics]:
        """
        Estimate question difficulty based on complexity analysis.
        
        Args:
            text: Question text to analyze
            topics: Optional pre-extracted topics
            
        Returns:
            Tuple of (difficulty_level, difficulty_score, complexity_metrics)
        """
        # Analyze complexity
        metrics = self.analyze_complexity(text)
        
        # Extract topics if not provided
        if topics is None:
            topics = self.extract_topics(text)
        
        # Topic-based difficulty adjustment
        topic_difficulty_boost = 0.0
        if topics:
            # Some topics are inherently more difficult
            difficult_topics = {'physics', 'chemistry', 'mathematics', 'technology'}
            topic_names = {t.topic_name for t in topics}
            
            if topic_names & difficult_topics:
                topic_difficulty_boost = 0.1
        
        # Calculate final difficulty score
        base_score = metrics.overall_complexity
        difficulty_score = min(1.0, base_score + topic_difficulty_boost)
        
        # Adjust for specific patterns
        if metrics.requires_calculation:
            difficulty_score = max(0.5, difficulty_score)  # Calculations are at least medium
        
        if metrics.requires_reasoning:
            difficulty_score = max(0.4, difficulty_score)  # Reasoning is at least medium-low
        
        # Map to difficulty level
        if difficulty_score < 0.25:
            level = DifficultyLevel.EASY
        elif difficulty_score < 0.5:
            level = DifficultyLevel.MEDIUM
        elif difficulty_score < 0.75:
            level = DifficultyLevel.HARD
        else:
            level = DifficultyLevel.EXPERT
        
        logfire.info(
            "Difficulty estimated",
            text_preview=text[:50],
            difficulty_level=level.value,
            difficulty_score=difficulty_score
        )
        
        return level, difficulty_score, metrics
    
    def classify_question_type(self, text: str) -> Dict[str, bool]:
        """
        Classify the type of question being asked.
        
        Args:
            text: Question text
            
        Returns:
            Dictionary of question type indicators
        """
        text_lower = text.lower()
        
        return {
            'factual': any(q in text_lower for q in ['what', 'when', 'where', 'who']),
            'explanatory': any(q in text_lower for q in ['why', 'how', 'explain']),
            'computational': any(q in text_lower for q in ['calculate', 'compute', 'solve']),
            'comparative': any(q in text_lower for q in ['compare', 'difference', 'similarity']),
            'evaluative': any(q in text_lower for q in ['evaluate', 'assess', 'judge']),
            'hypothetical': any(q in text_lower for q in ['if', 'would', 'suppose']),
            'definitional': any(q in text_lower for q in ['define', 'what is', 'meaning']),
            'procedural': any(q in text_lower for q in ['how to', 'steps', 'process']),
            'analytical': any(q in text_lower for q in ['analyze', 'examine', 'investigate']),
            'yes_no': text.strip().startswith(('is', 'are', 'do', 'does', 'can', 'will'))
        }


# Global instance for convenience
default_extractor = EntityExtractor()