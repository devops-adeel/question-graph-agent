#!/usr/bin/env python3
"""Test script to debug Answer creation issue."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from question_graph import Answer
from pydantic_graph import BaseNode
import inspect

print("Testing Answer creation...")

# Check BaseNode pattern
print(f"\nBaseNode __init__: {BaseNode.__init__}")
print(f"Answer MRO: {Answer.__mro__}")

# Try the dataclass-style creation
try:
    # Test creating via dataclass
    answer1 = Answer
    print(f"Answer class: {answer1}")
    print(f"Answer fields: {answer1.__fields__ if hasattr(answer1, '__fields__') else 'No fields'}")
    print(f"Answer annotations: {answer1.__annotations__ if hasattr(answer1, '__annotations__') else 'No annotations'}")
    
    # Try model_validate
    if hasattr(Answer, 'model_validate'):
        answer2 = Answer.model_validate({"question": "Test question?"})
        print(f"✓ model_validate works: {answer2.question}")
    
    # Try parse_obj
    if hasattr(Answer, 'parse_obj'):
        answer3 = Answer.parse_obj({"question": "Test question?"})
        print(f"✓ parse_obj works: {answer3.question}")
        
    # Try construct
    if hasattr(Answer, 'model_construct'):
        answer4 = Answer.model_construct(question="Test question?")
        print(f"✓ model_construct works: {answer4.question}")
        
except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()

# Look for the actual way to create instances
print("\nInspecting Answer methods:")
for name in dir(Answer):
    if not name.startswith('_') and callable(getattr(Answer, name)):
        print(f"  {name}: {getattr(Answer, name)}")