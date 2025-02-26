#!/usr/bin/env python3
"""
Test script for CSV functionality in BMW Agents.
This demonstrates reading and writing CSV files with the fixed CSV tools.
"""

import asyncio
import os
import sys
import tempfile

# Add parent directory to path so we can import from bmw_agents
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bmw_agents.core.toolbox.tools.file_tools import csv_read, csv_write
from bmw_agents.utils.logger import setup_logger, get_logger

# Set up logging
logger = get_logger(__name__)
setup_logger(level="INFO")

async def main():
    """Test CSV read and write functionality."""
    logger.info("Testing CSV read/write functionality...")
    
    # Create test data
    test_data = [
        {"name": "Alice", "age": "30", "city": "New York"},
        {"name": "Bob", "age": "25", "city": "San Francisco"},
        {"name": "Charlie", "age": "35", "city": "Chicago"}
    ]
    
    # Create a temporary file for testing
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as temp_file:
        temp_path = temp_file.name
    
    try:
        # Test writing CSV
        logger.info(f"Writing test data to CSV file: {temp_path}")
        csv_write(temp_path, test_data)
        
        # Test reading CSV
        logger.info(f"Reading test data from CSV file: {temp_path}")
        read_data = csv_read(temp_path)
        
        # Verify data
        logger.info(f"Read {len(read_data)} records from CSV file")
        for i, record in enumerate(read_data):
            logger.info(f"Record {i+1}: {record}")
            
        # Test mixed data (dict and list)
        mixed_data = [
            {"name": "Alice", "age": "30", "city": "New York"},
            ["Bob", "25", "San Francisco"],
            {"name": "Charlie", "age": "35", "city": "Chicago"}
        ]
        
        # Test writing mixed CSV
        logger.info(f"Writing mixed test data to CSV file: {temp_path}")
        csv_write(temp_path, mixed_data)
        
        # Test reading CSV
        logger.info(f"Reading mixed test data from CSV file: {temp_path}")
        read_mixed_data = csv_read(temp_path)
        
        # Verify data
        logger.info(f"Read {len(read_mixed_data)} records from mixed CSV file")
        for i, record in enumerate(read_mixed_data):
            logger.info(f"Record {i+1}: {record}")
            
        logger.info("CSV tests completed successfully!")
        
    finally:
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)
            logger.info(f"Removed temporary file: {temp_path}")

if __name__ == "__main__":
    asyncio.run(main()) 