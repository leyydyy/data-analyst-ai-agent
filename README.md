## Data Analyst AI Agent

The Data Analyst AI Agent is an autonomous Streamlit application designed to streamline the analytical workflow for students and non-technical users. It allows users to upload raw datasets, automatically evaluate data quality, propose and execute AI-generated cleaning strategies, and provide dynamic visualizations and insights.


<img width="1904" height="909" alt="image" src="https://github.com/user-attachments/assets/7cd290c9-a4a2-4111-87d1-b716b8bd590f" />



### This was built with:
- Streamlit
- OpenAI's gpt-4o and gpt-4o-mini
- Python Pandas for Data Processing
- Python Plotly Express for Visualization

### Installation Guide
1. **Clone the repository
```git clone https://github.com/lcandaya/data-analyst-ai-agent.git```
2. Install dependencies
```pip install -r requirements.txt```
3. Setup environmental variables by creating a .env file and add your OpenAI API Key
```OPENAI_API_KEY=your_api_key```
4. Run the application
```streanlit run app.py```

### How it Works
Upload a CSV or Excel file first via Streamlit, and ensure that any sensitive data is anonymized. The system extracts a lightweight metadata summary to save on token usage and preserve data privacy before evaluating the data quality. OpenAI's gpt-4o is used to map charts while gpt-4o-mini are used for the other features.

### Limitations of this Project
- The agent only reads a 5-row sample and dataset metadata to save on tokens. It cannot answer very specific questions about individual rows outside the sample.
- Cleaning strategies are limited. The system can only use predefined actions and highly complex cleaning strategies are out of its scope.
- Visualizations are restricted to predefined standard Plotly charts.
