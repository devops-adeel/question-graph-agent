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


# Sub-task 3.3: Answer classification models
class AnswerPattern(BaseModel):
    """Pattern for matching answer types."""
    
    pattern_type: str  # 'exact', 'fuzzy', 'semantic', 'numeric', 'pattern'
    expected_values: List[str] = Field(default_factory=list)
    similarity_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    case_sensitive: bool = False
    allow_partial: bool = True
    numeric_tolerance: Optional[float] = None


class AnswerEvaluation(BaseModel):
    """Result of answer evaluation."""
    
    status: AnswerStatus
    confidence: float = Field(ge=0.0, le=1.0)
    matched_patterns: List[str] = Field(default_factory=list)
    partial_matches: List[str] = Field(default_factory=list)
    feedback: str
    suggestions: List[str] = Field(default_factory=list)
    score: float = Field(ge=0.0, le=1.0)  # 0=incorrect, 0.5=partial, 1=correct


class AnswerClassifier:
    """Classify answers as correct, incorrect, or partial."""
    
    def __init__(self):
        """Initialize the answer classifier."""
        self.similarity_cache: Dict[Tuple[str, str], float] = {}
        self.pattern_cache: Dict[str, re.Pattern] = {}
    
    def calculate_similarity(self, text1: str, text2: str, method: str = 'token') -> float:
        """
        Calculate similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            method: Similarity method ('exact', 'token', 'fuzzy')
            
        Returns:
            Similarity score (0-1)
        """
        # Check cache
        cache_key = (text1.lower(), text2.lower())
        if cache_key in self.similarity_cache:
            return self.similarity_cache[cache_key]
        
        if method == 'exact':
            score = 1.0 if text1.lower() == text2.lower() else 0.0
        
        elif method == 'token':
            # Token-based similarity (Jaccard)
            tokens1 = set(re.findall(r'\b\w+\b', text1.lower()))
            tokens2 = set(re.findall(r'\b\w+\b', text2.lower()))
            
            if not tokens1 or not tokens2:
                score = 0.0
            else:
                intersection = tokens1 & tokens2
                union = tokens1 | tokens2
                score = len(intersection) / len(union)
        
        elif method == 'fuzzy':
            # Simple character-based similarity
            longer = max(len(text1), len(text2))
            if longer == 0:
                score = 1.0
            else:
                # Count matching characters in order
                matches = 0
                for i, char in enumerate(text1.lower()):
                    if i < len(text2) and char == text2.lower()[i]:
                        matches += 1
                score = matches / longer
        
        else:
            score = 0.0
        
        # Cache result
        self.similarity_cache[cache_key] = score
        return score
    
    def evaluate_numeric_answer(
        self,
        answer: str,
        expected: str,
        tolerance: float = 0.01
    ) -> Tuple[bool, float]:
        """
        Evaluate numeric answers with tolerance.
        
        Args:
            answer: User's answer
            expected: Expected answer
            tolerance: Acceptable deviation (default 1%)
            
        Returns:
            Tuple of (is_correct, confidence)
        """
        try:
            # Extract numbers from both strings
            answer_nums = re.findall(r'-?\d+\.?\d*', answer)
            expected_nums = re.findall(r'-?\d+\.?\d*', expected)
            
            if not answer_nums or not expected_nums:
                return False, 0.0
            
            # Convert to float
            answer_val = float(answer_nums[0])
            expected_val = float(expected_nums[0])
            
            # Check if within tolerance
            if expected_val == 0:
                is_correct = answer_val == 0
            else:
                relative_error = abs(answer_val - expected_val) / abs(expected_val)
                is_correct = relative_error <= tolerance
            
            # Calculate confidence based on accuracy
            if is_correct:
                if answer_val == expected_val:
                    confidence = 1.0
                else:
                    confidence = 1.0 - (relative_error / tolerance)
            else:
                confidence = max(0.0, 1.0 - relative_error)
            
            return is_correct, confidence
            
        except (ValueError, IndexError):
            return False, 0.0
    
    def classify_answer(
        self,
        answer: str,
        expected_patterns: List[AnswerPattern],
        question_type: Optional[Dict[str, bool]] = None
    ) -> AnswerEvaluation:
        """
        Classify an answer based on expected patterns.
        
        Args:
            answer: User's answer
            expected_patterns: List of acceptable answer patterns
            question_type: Optional question type classification
            
        Returns:
            AnswerEvaluation with status and details
        """
        answer = answer.strip()
        best_score = 0.0
        best_pattern = None
        all_matches = []
        partial_matches = []
        
        for pattern in expected_patterns:
            score = 0.0
            matched = False
            
            if pattern.pattern_type == 'exact':
                for expected in pattern.expected_values:
                    if pattern.case_sensitive:
                        if answer == expected:
                            score = 1.0
                            matched = True
                            break
                    else:
                        if answer.lower() == expected.lower():
                            score = 1.0
                            matched = True
                            break
            
            elif pattern.pattern_type == 'fuzzy':
                for expected in pattern.expected_values:
                    sim = self.calculate_similarity(answer, expected, 'token')
                    if sim >= pattern.similarity_threshold:
                        score = sim
                        matched = True
                        break
                    elif sim > 0.5 and pattern.allow_partial:
                        partial_matches.append(expected)
            
            elif pattern.pattern_type == 'numeric':
                for expected in pattern.expected_values:
                    is_correct, confidence = self.evaluate_numeric_answer(
                        answer, expected, 
                        pattern.numeric_tolerance or 0.01
                    )
                    if is_correct:
                        score = confidence
                        matched = True
                        break
            
            elif pattern.pattern_type == 'pattern':
                for expected_pattern in pattern.expected_values:
                    if expected_pattern not in self.pattern_cache:
                        self.pattern_cache[expected_pattern] = re.compile(
                            expected_pattern, 
                            re.IGNORECASE if not pattern.case_sensitive else 0
                        )
                    
                    regex = self.pattern_cache[expected_pattern]
                    if regex.search(answer):
                        score = 1.0
                        matched = True
                        break
            
            if matched:
                all_matches.append(pattern.expected_values[0])
            
            if score > best_score:
                best_score = score
                best_pattern = pattern
        
        # Determine status and feedback
        if best_score >= 0.9:
            status = AnswerStatus.CORRECT
            feedback = "Excellent! Your answer is correct."
            suggestions = []
        elif best_score >= 0.6 and best_pattern and best_pattern.allow_partial:
            status = AnswerStatus.PARTIAL
            feedback = "Good effort! Your answer is partially correct."
            suggestions = ["Consider being more specific", "Review the exact terminology"]
        else:
            status = AnswerStatus.INCORRECT
            feedback = "Not quite right. Let's review this topic."
            suggestions = self._generate_suggestions(answer, expected_patterns, partial_matches)
        
        # Adjust for yes/no questions
        if question_type and question_type.get('yes_no'):
            if self._is_yes_no_answer(answer):
                # For yes/no questions, partial credit doesn't make sense
                if best_score > 0.5:
                    status = AnswerStatus.CORRECT
                    best_score = 1.0
                else:
                    status = AnswerStatus.INCORRECT
                    best_score = 0.0
        
        return AnswerEvaluation(
            status=status,
            confidence=best_score,
            matched_patterns=all_matches,
            partial_matches=partial_matches,
            feedback=feedback,
            suggestions=suggestions,
            score=best_score
        )
    
    def _is_yes_no_answer(self, answer: str) -> bool:
        """Check if answer is a yes/no response."""
        answer_lower = answer.lower().strip()
        yes_patterns = {'yes', 'y', 'yeah', 'yep', 'correct', 'true', 'affirmative'}
        no_patterns = {'no', 'n', 'nope', 'incorrect', 'false', 'negative'}
        
        return answer_lower in yes_patterns or answer_lower in no_patterns
    
    def _generate_suggestions(
        self,
        answer: str,
        expected_patterns: List[AnswerPattern],
        partial_matches: List[str]
    ) -> List[str]:
        """Generate helpful suggestions for incorrect answers."""
        suggestions = []
        
        # Check if answer is too short
        if len(answer.split()) < 3:
            suggestions.append("Try providing more detail in your answer")
        
        # Check for common mistakes
        if partial_matches:
            suggestions.append(f"You're on the right track. Consider: {partial_matches[0]}")
        
        # Check for numeric patterns
        has_numbers = re.search(r'\d+', answer)
        expects_numbers = any(
            p.pattern_type == 'numeric' 
            for p in expected_patterns
        )
        
        if expects_numbers and not has_numbers:
            suggestions.append("This question expects a numeric answer")
        elif has_numbers and not expects_numbers:
            suggestions.append("Check if a numeric answer is appropriate here")
        
        # Limit suggestions
        return suggestions[:3]
    
    def create_answer_patterns(
        self,
        correct_answers: List[str],
        question_type: Optional[Dict[str, bool]] = None
    ) -> List[AnswerPattern]:
        """
        Create answer patterns from correct answer examples.
        
        Args:
            correct_answers: List of acceptable answers
            question_type: Optional question type classification
            
        Returns:
            List of AnswerPattern objects
        """
        patterns = []
        
        # Analyze answer characteristics
        all_numeric = all(
            re.match(r'^-?\d+\.?\d*\s*\w*$', ans.strip()) 
            for ans in correct_answers
        )
        
        if all_numeric:
            # Numeric answers
            patterns.append(AnswerPattern(
                pattern_type='numeric',
                expected_values=correct_answers,
                numeric_tolerance=0.01,
                allow_partial=False
            ))
        
        elif question_type and question_type.get('yes_no'):
            # Yes/no questions
            patterns.append(AnswerPattern(
                pattern_type='exact',
                expected_values=correct_answers,
                case_sensitive=False,
                allow_partial=False
            ))
        
        else:
            # Text answers - use multiple strategies
            # Exact match
            patterns.append(AnswerPattern(
                pattern_type='exact',
                expected_values=correct_answers,
                case_sensitive=False,
                allow_partial=False
            ))
            
            # Fuzzy match for longer answers
            if any(len(ans.split()) > 3 for ans in correct_answers):
                patterns.append(AnswerPattern(
                    pattern_type='fuzzy',
                    expected_values=correct_answers,
                    similarity_threshold=0.7,
                    allow_partial=True
                ))
        
        return patterns


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
        
        # Initialize answer classifier
        self.answer_classifier = AnswerClassifier()
    
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
    
    # Sub-task 3.3: Answer classification methods
    def extract_answer_entity(
        self,
        answer_text: str,
        question_text: str,
        expected_answers: Optional[List[str]] = None,
        user_id: str = "anonymous",
        session_id: str = "default",
        response_time: float = 0.0
    ) -> Tuple[AnswerEntity, AnswerEvaluation]:
        """
        Extract an AnswerEntity with evaluation from answer text.
        
        Args:
            answer_text: User's answer
            question_text: The question being answered
            expected_answers: Optional list of correct answers
            user_id: User identifier
            session_id: Session identifier
            response_time: Time taken to answer
            
        Returns:
            Tuple of (AnswerEntity, AnswerEvaluation)
        """
        # Get question type for better classification
        question_type = self.classify_question_type(question_text)
        
        # Create answer patterns if expected answers provided
        if expected_answers:
            patterns = self.answer_classifier.create_answer_patterns(
                expected_answers,
                question_type
            )
            
            # Evaluate the answer
            evaluation = self.answer_classifier.classify_answer(
                answer_text,
                patterns,
                question_type
            )
        else:
            # No expected answers - create unevaluated response
            evaluation = AnswerEvaluation(
                status=AnswerStatus.UNEVALUATED,
                confidence=0.0,
                feedback="Answer recorded but not evaluated",
                score=0.0
            )
        
        # Create answer entity
        answer_entity = AnswerEntity(
            question_id=f"q_{hash(question_text)}",
            user_id=user_id,
            session_id=session_id,
            content=answer_text,
            status=evaluation.status,
            confidence_score=evaluation.confidence,
            response_time_seconds=response_time,
            feedback=evaluation.feedback
        )
        
        logfire.info(
            "Answer entity extracted",
            status=evaluation.status.value,
            confidence=evaluation.confidence,
            has_suggestions=len(evaluation.suggestions) > 0
        )
        
        return answer_entity, evaluation
    
    def evaluate_answer_similarity(
        self,
        answer1: str,
        answer2: str,
        method: str = 'token'
    ) -> float:
        """
        Evaluate similarity between two answers.
        
        Args:
            answer1: First answer
            answer2: Second answer
            method: Similarity method ('exact', 'token', 'fuzzy')
            
        Returns:
            Similarity score (0-1)
        """
        return self.answer_classifier.calculate_similarity(answer1, answer2, method)
    
    def create_answer_patterns_from_qa(
        self,
        question: str,
        correct_answers: List[str],
        topic_hints: Optional[List[str]] = None
    ) -> List[AnswerPattern]:
        """
        Create intelligent answer patterns based on question and answers.
        
        Args:
            question: Question text
            correct_answers: List of correct answers
            topic_hints: Optional topic hints for pattern creation
            
        Returns:
            List of AnswerPattern objects
        """
        question_type = self.classify_question_type(question)
        patterns = self.answer_classifier.create_answer_patterns(
            correct_answers,
            question_type
        )
        
        # Enhance patterns based on topics if provided
        if topic_hints:
            # Add pattern variations based on topic terminology
            for topic in topic_hints:
                if topic in self.topics:
                    topic_def = self.topics[topic]
                    
                    # Check if any keywords should be acceptable alternatives
                    for keyword in topic_def.primary_keywords:
                        for pattern in patterns:
                            if pattern.pattern_type == 'fuzzy':
                                # Add topic keywords as acceptable variations
                                pattern.expected_values.extend([
                                    ans.replace(word, keyword)
                                    for ans in correct_answers
                                    for word in ans.split()
                                    if word.lower() in topic_def.secondary_keywords
                                ])
        
        return patterns
    
    def generate_answer_feedback(
        self,
        evaluation: AnswerEvaluation,
        question_difficulty: Optional[DifficultyLevel] = None,
        user_mastery: Optional[float] = None
    ) -> str:
        """
        Generate personalized feedback based on evaluation and context.
        
        Args:
            evaluation: Answer evaluation result
            question_difficulty: Optional question difficulty
            user_mastery: Optional user mastery level (0-1)
            
        Returns:
            Personalized feedback message
        """
        base_feedback = evaluation.feedback
        
        # Enhance feedback based on context
        if evaluation.status == AnswerStatus.CORRECT:
            if question_difficulty == DifficultyLevel.EXPERT:
                return f"{base_feedback} Impressive work on this challenging question!"
            elif user_mastery and user_mastery < 0.5:
                return f"{base_feedback} You're making great progress!"
                
        elif evaluation.status == AnswerStatus.PARTIAL:
            if evaluation.suggestions:
                return f"{base_feedback}\n\nHints: {'; '.join(evaluation.suggestions[:2])}"
            else:
                return f"{base_feedback} You're close to the complete answer."
                
        elif evaluation.status == AnswerStatus.INCORRECT:
            if question_difficulty == DifficultyLevel.EASY and user_mastery and user_mastery > 0.7:
                return "This seems like a simple mistake. Take another look at the question."
            elif evaluation.partial_matches:
                return f"{base_feedback} You mentioned some relevant concepts: {', '.join(evaluation.partial_matches[:2])}"
        
        return base_feedback
    
    def analyze_answer_patterns(
        self,
        answer_history: List[Tuple[str, AnswerStatus]]
    ) -> Dict[str, Any]:
        """
        Analyze patterns in answer history.
        
        Args:
            answer_history: List of (answer_text, status) tuples
            
        Returns:
            Dictionary with pattern analysis
        """
        if not answer_history:
            return {
                'total_answers': 0,
                'accuracy_rate': 0.0,
                'common_mistakes': [],
                'improvement_trend': 'insufficient_data'
            }
        
        # Basic statistics
        total = len(answer_history)
        correct = sum(1 for _, status in answer_history if status == AnswerStatus.CORRECT)
        partial = sum(1 for _, status in answer_history if status == AnswerStatus.PARTIAL)
        
        # Common mistakes (for incorrect answers)
        incorrect_answers = [ans for ans, status in answer_history if status == AnswerStatus.INCORRECT]
        
        # Find common patterns in mistakes
        common_mistakes = []
        if incorrect_answers:
            # Look for common words in incorrect answers
            word_freq = Counter()
            for ans in incorrect_answers:
                words = re.findall(r'\b\w+\b', ans.lower())
                word_freq.update(words)
            
            # Filter out common words
            stop_words = {'the', 'a', 'an', 'is', 'it', 'and', 'or', 'but', 'in', 'on', 'at'}
            common_mistakes = [
                word for word, count in word_freq.most_common(5)
                if count > 1 and word not in stop_words
            ]
        
        # Improvement trend (simple)
        if total < 3:
            trend = 'insufficient_data'
        else:
            recent_accuracy = sum(
                1 for _, status in answer_history[-3:]
                if status in [AnswerStatus.CORRECT, AnswerStatus.PARTIAL]
            ) / 3
            
            early_accuracy = sum(
                1 for _, status in answer_history[:3]
                if status in [AnswerStatus.CORRECT, AnswerStatus.PARTIAL]
            ) / 3
            
            if recent_accuracy > early_accuracy + 0.2:
                trend = 'improving'
            elif recent_accuracy < early_accuracy - 0.2:
                trend = 'declining'
            else:
                trend = 'stable'
        
        return {
            'total_answers': total,
            'correct_answers': correct,
            'partial_answers': partial,
            'accuracy_rate': correct / total,
            'partial_credit_rate': (correct + partial * 0.5) / total,
            'common_mistakes': common_mistakes,
            'improvement_trend': trend
        }


# Sub-task 3.5: Async NLP integration points
class NLPModel(BaseModel):
    """Base class for NLP model integration."""
    
    model_name: str = Field(..., description="Name/identifier of the NLP model")
    model_type: str = Field(..., description="Type of model (e.g., 'embedding', 'ner', 'classification')")
    is_loaded: bool = Field(default=False, description="Whether the model is loaded")
    config: Dict[str, Any] = Field(default_factory=dict, description="Model configuration")
    
    async def load(self) -> None:
        """Load the NLP model asynchronously."""
        raise NotImplementedError("Subclasses must implement load()")
    
    async def predict(self, text: str) -> Any:
        """Make predictions on text."""
        raise NotImplementedError("Subclasses must implement predict()")


class TopicExtractionModel(NLPModel):
    """Placeholder for advanced topic extraction using NLP models."""
    
    def __init__(self):
        super().__init__(
            model_name="topic_extraction_placeholder",
            model_type="classification"
        )
    
    async def load(self) -> None:
        """Load topic extraction model (placeholder)."""
        # In production, this would load a model like:
        # - BERT-based topic classifier
        # - LDA topic model
        # - Custom embedding model
        self.is_loaded = True
        logfire.info("Topic extraction model loaded (placeholder)")
    
    async def predict(self, text: str) -> List[Tuple[str, float]]:
        """
        Predict topics from text.
        
        Returns:
            List of (topic_name, confidence) tuples
        """
        # Placeholder: return empty list
        # In production, this would use the loaded model
        return []


class DifficultyEstimationModel(NLPModel):
    """Placeholder for ML-based difficulty estimation."""
    
    def __init__(self):
        super().__init__(
            model_name="difficulty_estimation_placeholder",
            model_type="regression"
        )
    
    async def load(self) -> None:
        """Load difficulty estimation model (placeholder)."""
        # In production, this would load:
        # - Readability models
        # - Complexity prediction models
        # - Fine-tuned language models
        self.is_loaded = True
        logfire.info("Difficulty estimation model loaded (placeholder)")
    
    async def predict(self, text: str) -> float:
        """
        Predict difficulty score from text.
        
        Returns:
            Difficulty score (0-1)
        """
        # Placeholder: return medium difficulty
        return 0.5


class AnswerSimilarityModel(NLPModel):
    """Placeholder for semantic similarity matching."""
    
    def __init__(self):
        super().__init__(
            model_name="answer_similarity_placeholder",
            model_type="embedding"
        )
        self.embeddings_cache: Dict[str, Any] = {}
    
    async def load(self) -> None:
        """Load similarity model (placeholder)."""
        # In production, this would load:
        # - Sentence transformers
        # - Word2Vec/GloVe models
        # - Custom embedding models
        self.is_loaded = True
        logfire.info("Answer similarity model loaded (placeholder)")
    
    async def encode(self, texts: List[str]) -> Any:
        """
        Encode texts to embeddings.
        
        Args:
            texts: List of texts to encode
            
        Returns:
            Embeddings (placeholder returns None)
        """
        # Placeholder implementation
        for text in texts:
            if text not in self.embeddings_cache:
                self.embeddings_cache[text] = None
        return None
    
    async def similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity between texts.
        
        Returns:
            Similarity score (0-1)
        """
        # Placeholder: return basic string similarity
        return 0.5


class AsyncEntityExtractor(EntityExtractor):
    """
    Enhanced entity extractor with async NLP model support.
    
    This class extends EntityExtractor to provide async methods that can
    leverage NLP models when available, falling back to keyword-based
    extraction when models are not loaded.
    """
    
    def __init__(
        self,
        custom_topics: Optional[List[TopicKeywords]] = None,
        enable_nlp: bool = False
    ):
        """
        Initialize async entity extractor.
        
        Args:
            custom_topics: Custom topic definitions
            enable_nlp: Whether to enable NLP model integration
        """
        super().__init__(custom_topics)
        self.enable_nlp = enable_nlp
        
        # Initialize NLP models (not loaded by default)
        self.topic_model = TopicExtractionModel()
        self.difficulty_model = DifficultyEstimationModel()
        self.similarity_model = AnswerSimilarityModel()
        
        self._models_loaded = False
    
    async def load_nlp_models(self) -> None:
        """Load all NLP models asynchronously."""
        if not self.enable_nlp:
            logfire.warning("NLP models not enabled")
            return
        
        with logfire.span("async_extractor.load_nlp_models") as span:
            try:
                # Load models in parallel
                await asyncio.gather(
                    self.topic_model.load(),
                    self.difficulty_model.load(),
                    self.similarity_model.load()
                )
                self._models_loaded = True
                span.set_attribute("models_loaded", True)
                logfire.info("All NLP models loaded successfully")
            except Exception as e:
                span.set_attribute("error", str(e))
                logfire.error("Failed to load NLP models", error=str(e))
                self._models_loaded = False
    
    async def extract_topics_async(
        self,
        text: str,
        min_confidence: float = 0.3,
        max_topics: int = 5,
        use_nlp: bool = True
    ) -> List[ExtractedTopic]:
        """
        Extract topics asynchronously with optional NLP enhancement.
        
        Args:
            text: Text to extract topics from
            min_confidence: Minimum confidence threshold
            max_topics: Maximum topics to return
            use_nlp: Whether to use NLP models if available
            
        Returns:
            List of extracted topics
        """
        with logfire.span("async_extractor.extract_topics") as span:
            span.set_attribute("text_length", len(text))
            span.set_attribute("use_nlp", use_nlp and self._models_loaded)
            
            # Start with keyword-based extraction
            keyword_topics = await asyncio.to_thread(
                self.extract_topics,
                text,
                min_confidence,
                max_topics
            )
            
            # Enhance with NLP if available
            if use_nlp and self._models_loaded and self.topic_model.is_loaded:
                try:
                    # Get NLP predictions
                    nlp_predictions = await self.topic_model.predict(text)
                    
                    # Merge results
                    topic_scores = {t.topic_name: t.confidence for t in keyword_topics}
                    
                    for topic_name, nlp_confidence in nlp_predictions:
                        if topic_name in topic_scores:
                            # Average scores if topic found by both methods
                            topic_scores[topic_name] = (
                                topic_scores[topic_name] + nlp_confidence
                            ) / 2
                        else:
                            # Add new topic from NLP
                            topic_scores[topic_name] = nlp_confidence
                    
                    # Convert back to ExtractedTopic objects
                    merged_topics = [
                        ExtractedTopic(
                            topic_name=name,
                            confidence=score,
                            match_type="hybrid"
                        )
                        for name, score in topic_scores.items()
                        if score >= min_confidence
                    ]
                    
                    # Sort and limit
                    merged_topics.sort(key=lambda x: x.confidence, reverse=True)
                    return merged_topics[:max_topics]
                    
                except Exception as e:
                    logfire.warning("NLP topic extraction failed", error=str(e))
                    # Fall back to keyword results
            
            return keyword_topics
    
    async def estimate_difficulty_async(
        self,
        text: str,
        topics: Optional[List[ExtractedTopic]] = None,
        use_nlp: bool = True
    ) -> Tuple[DifficultyLevel, float, ComplexityMetrics]:
        """
        Estimate difficulty asynchronously with optional NLP enhancement.
        
        Args:
            text: Question text
            topics: Pre-extracted topics
            use_nlp: Whether to use NLP models if available
            
        Returns:
            Tuple of (level, score, metrics)
        """
        with logfire.span("async_extractor.estimate_difficulty") as span:
            span.set_attribute("use_nlp", use_nlp and self._models_loaded)
            
            # Get keyword-based estimation
            level, score, metrics = await asyncio.to_thread(
                self.estimate_difficulty,
                text,
                topics
            )
            
            # Enhance with NLP if available
            if use_nlp and self._models_loaded and self.difficulty_model.is_loaded:
                try:
                    # Get NLP difficulty prediction
                    nlp_score = await self.difficulty_model.predict(text)
                    
                    # Combine scores (weighted average)
                    # Give more weight to metrics-based score as it's more explainable
                    combined_score = (score * 0.6 + nlp_score * 0.4)
                    
                    # Update level based on combined score
                    combined_level = DifficultyLevel.from_score(combined_score)
                    
                    span.set_attribute("nlp_score", nlp_score)
                    span.set_attribute("combined_score", combined_score)
                    
                    return combined_level, combined_score, metrics
                    
                except Exception as e:
                    logfire.warning("NLP difficulty estimation failed", error=str(e))
            
            return level, score, metrics
    
    async def classify_answer_async(
        self,
        answer: str,
        expected_patterns: List[AnswerPattern],
        question_type: Optional[Dict[str, bool]] = None,
        use_nlp: bool = True
    ) -> AnswerEvaluation:
        """
        Classify answer asynchronously with optional semantic similarity.
        
        Args:
            answer: User's answer
            expected_patterns: Expected answer patterns
            question_type: Question type info
            use_nlp: Whether to use NLP models if available
            
        Returns:
            AnswerEvaluation with enhanced similarity matching
        """
        with logfire.span("async_extractor.classify_answer") as span:
            span.set_attribute("use_nlp", use_nlp and self._models_loaded)
            
            # Get keyword-based classification
            evaluation = await asyncio.to_thread(
                self.answer_classifier.classify_answer,
                answer,
                expected_patterns,
                question_type
            )
            
            # Enhance with semantic similarity if available
            if (use_nlp and self._models_loaded and 
                self.similarity_model.is_loaded and 
                evaluation.status != AnswerStatus.CORRECT):
                
                try:
                    # Check semantic similarity for fuzzy patterns
                    for pattern in expected_patterns:
                        if pattern.pattern_type == 'fuzzy':
                            for expected in pattern.expected_values:
                                # Get semantic similarity
                                sem_score = await self.similarity_model.similarity(
                                    answer,
                                    expected
                                )
                                
                                # If high semantic similarity, upgrade evaluation
                                if sem_score > 0.8 and evaluation.status == AnswerStatus.INCORRECT:
                                    evaluation.status = AnswerStatus.PARTIAL
                                    evaluation.score = max(evaluation.score, sem_score * 0.7)
                                    evaluation.feedback = "Your answer is semantically similar to the expected response."
                                    evaluation.matched_patterns.append(f"semantic:{expected}")
                                    
                                    span.set_attribute("semantic_match", True)
                                    span.set_attribute("semantic_score", sem_score)
                                    break
                                    
                except Exception as e:
                    logfire.warning("Semantic similarity check failed", error=str(e))
            
            return evaluation
    
    async def process_qa_async(
        self,
        question_text: str,
        answer_text: str,
        expected_answers: Optional[List[str]] = None,
        user_id: str = "anonymous",
        session_id: str = "default"
    ) -> Dict[str, Any]:
        """
        Process complete Q&A interaction asynchronously.
        
        Args:
            question_text: Question text
            answer_text: User's answer
            expected_answers: Expected correct answers
            user_id: User identifier
            session_id: Session identifier
            
        Returns:
            Dictionary with extracted entities and evaluation
        """
        with logfire.span("async_extractor.process_qa") as span:
            # Process in parallel where possible
            tasks = []
            
            # Extract topics from question
            topics_task = asyncio.create_task(
                self.extract_topics_async(question_text)
            )
            tasks.append(topics_task)
            
            # Estimate difficulty
            difficulty_task = asyncio.create_task(
                self.estimate_difficulty_async(question_text)
            )
            tasks.append(difficulty_task)
            
            # Wait for parallel tasks
            topics = await topics_task
            difficulty_level, difficulty_score, metrics = await difficulty_task
            
            # Create question entity
            question_entity = await asyncio.to_thread(
                self.extract_entities_from_question,
                question_text,
                f"q_async_{hash(question_text)}"
            )
            question_entity.difficulty = difficulty_level
            question_entity.topics = [t.topic_name for t in topics]
            
            # Create answer patterns if expected answers provided
            if expected_answers:
                patterns = await asyncio.to_thread(
                    self.answer_classifier.create_answer_patterns,
                    expected_answers,
                    self.classify_question_type(question_text)
                )
                
                # Evaluate answer
                evaluation = await self.classify_answer_async(
                    answer_text,
                    patterns
                )
            else:
                evaluation = AnswerEvaluation(
                    status=AnswerStatus.UNEVALUATED,
                    confidence=0.0,
                    feedback="No expected answers provided",
                    score=0.0
                )
            
            # Create answer entity
            answer_entity = AnswerEntity(
                question_id=question_entity.id,
                user_id=user_id,
                session_id=session_id,
                content=answer_text,
                status=evaluation.status,
                confidence_score=evaluation.confidence,
                response_time_seconds=1.0,  # Would be tracked in practice
                feedback=evaluation.feedback
            )
            
            return {
                'question': question_entity,
                'answer': answer_entity,
                'topics': topics,
                'difficulty': {
                    'level': difficulty_level,
                    'score': difficulty_score,
                    'metrics': metrics
                },
                'evaluation': evaluation
            }


# Enhanced extraction pipeline with async support
class AsyncExtractionPipeline(ExtractionPipeline):
    """Extraction pipeline with async NLP support."""
    
    def __init__(
        self,
        extractor: Optional[AsyncEntityExtractor] = None,
        max_workers: int = 4,
        enable_nlp: bool = False
    ):
        """
        Initialize async extraction pipeline.
        
        Args:
            extractor: Async entity extractor instance
            max_workers: Maximum parallel workers
            enable_nlp: Whether to enable NLP models
        """
        self.async_extractor = extractor or AsyncEntityExtractor(enable_nlp=enable_nlp)
        super().__init__(self.async_extractor, max_workers, use_async=True)
        self.nlp_enabled = enable_nlp
    
    async def initialize(self):
        """Initialize the pipeline and load NLP models if enabled."""
        if self.nlp_enabled:
            await self.async_extractor.load_nlp_models()
    
    async def extract_with_nlp(
        self,
        texts: List[str],
        batch_size: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Extract entities from texts using NLP models.
        
        Args:
            texts: List of texts to process
            batch_size: Processing batch size
            
        Returns:
            List of extraction results
        """
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            # Process batch in parallel
            batch_tasks = [
                self.async_extractor.extract_topics_async(text)
                for text in batch
            ]
            
            batch_results = await asyncio.gather(*batch_tasks)
            results.extend(batch_results)
        
        return results


# Global instance for convenience
default_extractor = EntityExtractor()

# Async extractor instance (not initialized by default)
async_extractor = AsyncEntityExtractor(enable_nlp=False)