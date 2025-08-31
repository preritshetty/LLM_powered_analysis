# Flight Data Analysis and Cleaning Pipeline

## Project Overview
This project is a sophisticated data cleaning and analysis pipeline built using Python, Streamlit, and LangChain. It consists of two main components:
1. **Data Cleaner**: A three-phase cleaning pipeline that uses AI to detect and fix data quality issues
2. **Data Analyzer**: An AI-powered analysis tool that helps explore and visualize the cleaned data

## Key Features

### Data Cleaner
- **Phase 1**: Basic Data Cleaning
  - Handles missing values
  - Standardizes data formats
  - Performs initial data validation
  
- **Phase 2**: AI-Powered Issue Detection
  - Uses GPT-4.0 for intelligent data quality analysis
  - Detects patterns and anomalies
  - Provides data quality scoring
  - Sample size: 150 rows for analysis
  - Token limit: 200 tokens for efficient processing
  
- **Phase 3**: Automated Code Generation & Fixes
  - AI-generated Python code for data fixes
  - Automatic column name standardization
  - Intelligent data type corrections
  - Built-in execution engine for applying fixes

### Data Analyzer
- Interactive data exploration
- Natural language query interface
- Automatic date/time processing
- Dynamic visualization generation
- Column type detection (numerical, categorical, datetime)

## Setup Instructions

### Prerequisites
```bash
# Python version
Python 3.8 or higher

# Required environment variables
OPENAI_API_KEY=your_api_key_here
```

### Installation
1. Clone the repository:
```bash
git clone https://github.com/preritshetty/LLM_powered_analysis.git
cd flight_analysis_app
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
   - Create a `.env` file in the root directory
   - Add your OpenAI API key:
     ```
     OPENAI_API_KEY=your_api_key_here
     ```

### Running the Application
Run the application with a single command:
```bash
streamlit run cleaner/main.py
```

This launches the complete application. The analyzer component is integrated into the main application flow and will be automatically called when needed 


## Using the Pipeline

### Data Cleaning Process

1. **Phase 1**: Upload your raw data
   - Supports CSV files
   - Automatic basic cleaning
   - View cleaning reports

2. **Phase 2**: AI Analysis
   - Automatic issue detection
   - Data quality scoring
   - No configuration needed (uses optimized settings)

3. **Phase 3**: Code Generation
   - Review suggested column renamings
   - Auto-generated fixes
   - Final cleaned dataset saved as `data/final_cleaned.csv`

### Data Analysis Features

1. Load cleaned data (automatically reads from `data/final_cleaned.csv`)
2. Use natural language queries to analyze data
3. View automatic visualizations
4. Export results and insights

## Project Structure
```
flight_analysis_app/
├── analyzer/              # Data analysis module
│   ├── app.py            # Main analyzer application
│   └── modules/          # Analysis components
│       ├── data_loader.py    # Generic data loading
│       ├── llm_agent.py      # LLM interaction
│       └── query_handler.py   # Query processing
├── cleaner/              # Data cleaning module
│   ├── main.py          # Main cleaner application
│   ├── pipeline/        # Cleaning pipeline phases
│   │   ├── phase2_ai.py     # AI analysis
│   │   └── phase3_codegen.py # Code generation
│   └── utils/           # Utility functions
├── data/                # Data directory
│   └── final_cleaned.csv    # Cleaned output
└── requirements.txt     # Project dependencies
```

## Important Implementation Details

### AI Model Configuration
- Model: GPT-3.5-turbo or GPT-4 (optimized for cost and performance)
- Temperature: 0.1 (for consistent output)
- Max tokens: 200 (balanced for efficiency)
- Sample size: 150 rows for analysis

### Key Technical Features
1. **Generic Data Loading**
   - Automatic date/time detection
   - Smart column type inference
   - Robust error handling

2. **AI-Powered Analysis**
   - Pattern detection
   - Anomaly identification
   - Quality scoring
   - Natural language interface

3. **Code Generation**
   - Python code generation for fixes
   - Safe code execution
   - Automated testing

4. **Data Processing**
   - Column standardization
   - Type conversion
   - Missing value handling
   - Date/time parsing

### Error Handling & Logging
- Comprehensive error capture
- Detailed execution logs
- User-friendly error messages

### Performance Optimizations
- Efficient data sampling
- Optimized token usage
- Parallel processing where possible

## Testing Instructions

1. **Initial Setup Test**
   ```bash
   # Verify environment
   python -c "import pandas, streamlit, langchain"
   
   # Check OpenAI API
   python -c "import os; print(os.getenv('OPENAI_API_KEY'))"
   ```

2. **Pipeline Test**
   ```python
   # Sample data structure
   data = {
       'Date_Column': ['2023-01-01', '2023-01-02'],
       'Category_Column': ['A', 'B'],
       'Value_Column': [100, 150]
   }
   ```

3. **Verification Steps**
   - Check cleaned output in `data/final_cleaned.csv`
   - Verify column standardization
   - Test analyzer queries

## Limitations

1. **Data Size**
   - Optimal: < 1M rows
   - Uses sampling for larger datasets

2. **API Dependencies**
   - Requires OpenAI API key
   - Internet connection needed

3. **Processing Time**
   - Phase 2: ~30 seconds
   - Phase 3: ~1 minute

## Future Enhancements

1. Additional data formats support
2. Custom cleaning rules
3. Advanced visualization options
4. Batch processing
5. Extended model support

## Repository
- Project: https://github.com/preritshetty/LLM_powered_analysis.git

# LLM_powered_analysis
