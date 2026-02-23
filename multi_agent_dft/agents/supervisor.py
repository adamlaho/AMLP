"""
Supervisor agents for the Multi-Agent DFT Research System.
"""

import logging
from .base import Agent
from ..api.publication import PublicationAPI
import re

logger = logging.getLogger(__name__)


class SupervisorAgent(Agent):
    """Agent to integrate and provide recommendations."""
    
    def __init__(self, role="Integration", name=None, model=None, config=None):
        """
        Initialize the supervisor agent.
        
        Args:
            role (str, optional): Specific role of this supervisor.
            name (str, optional): Agent name. If None, will be constructed from role.
            model (str, optional): OpenAI model to use.
            config (dict, optional): Configuration dictionary.
        """
        if name is None:
            name = f"SupervisorAgent_{role}"
        
        super().__init__(name, model, config)
        self.role = role
        self.config = config or {}
        
        # Initialize PublicationAPI with config
        from multi_agent_dft.api.publication import PublicationAPI
        self.pub_api = PublicationAPI(config=self.config)
        
        # Use premium model for supervisor if available
        premium_model = self.config.get('agents', {}).get('models', {}).get('premium')
        if premium_model:
            self.model = premium_model
    
    def get_system_prompt(self):
        """
        Get the system prompt for the supervisor agent.
        
        Returns:
            str: Specialized system prompt based on the supervisor's role.
        """
        base_prompt = f"You are {self.name}, a supervisor agent in the Multi-Agent DFT Research System."
        
        if self.role == "Integration":
            return (
                f"{base_prompt} Your role is to integrate information from different expert agents, "
                "identify patterns and connections, and provide a comprehensive synthesis. "
                "Balance experimental and theoretical perspectives, highlight consensus findings, "
                "note contradictions, and identify knowledge gaps. Your goal is to create a cohesive "
                "overview that combines diverse expertise into actionable insights."
            )
        elif self.role == "DFT_Recommendation":
            return (
                f"{base_prompt} Your role is to evaluate input from multiple DFT expert agents "
                "and provide specific software and parameter recommendations for the research question. "
                "Consider the strengths and limitations of each DFT code, the specific requirements of the "
                "chemical system, computational efficiency, and accuracy requirements. Provide clear, "
                "actionable recommendations with justification."
            )
        else:
            return base_prompt
    
    def integrate(self, content, additional_context=""):
        """
        Integrate information from multiple sources and provide a synthesis.
        
        Args:
            content (str): Content from multiple sources to integrate.
            additional_context (str, optional): Additional context to consider.
        
        Returns:
            str: Integrated synthesis and recommendations.
        """
        system_content = self.get_system_prompt()
        user_content = f"Please integrate and synthesize the following information:\n\n{content}"
        
        if additional_context:
            user_content += f"\n\nAdditional context to consider:\n{additional_context}"
        
        return self.query(user_content, system_content)
    
    def generate_followup_question(self, research_query, max_results=5):
        """
        Generate a follow-up question based on an initial search.
        
        Args:
            research_query (str): Initial research query.
            max_results (int, optional): Maximum number of publications to analyze.
        
        Returns:
            str: Follow-up question to refine the research focus.
        """
        # Search for publications related to the query
        publications = self.pub_api.search(research_query, max_results=max_results)
        
        if not publications:
            return "Could you provide more details about your research focus and specific questions?"
        
        # Analyze publications to identify key areas
        analysis = self.pub_api.analyze_publications(publications)
        
        # Extract key topics from keywords, titles, and abstracts
        topics = []
        
        # From keywords
        for keyword, count in analysis.get('keyword_analysis', {}).get('most_common', []):
            if count > 1:  # Only include keywords mentioned multiple times
                topics.append(keyword)
        
        # From titles and abstracts
        title_abstract_text = ""
        for pub in publications[:3]:  # Look at top 3 publications
            title_abstract_text += f"{pub.get('title', '')} {pub.get('abstract', '')}"
        
        # Extract phrases that might be research aspects (2-3 word phrases)
        import re
        phrases = re.findall(r'\b([A-Za-z][\w-]*(?:\s+[A-Za-z][\w-]*){1,2})\b', title_abstract_text.lower())
        phrase_count = {}
        for phrase in phrases:
            if len(phrase) > 5:  # Filter out very short phrases
                phrase_count[phrase] = phrase_count.get(phrase, 0) + 1
        
        # Add top phrases to topics
        for phrase, count in sorted(phrase_count.items(), key=lambda x: x[1], reverse=True)[:5]:
            topics.append(phrase)
        
        # Generate follow-up question based on identified topics
        system_content = (
            f"You are {self.name}, a research supervisor. Based on the initial search results "
            "for a research query, generate a thoughtful follow-up question that will help refine "
            "the research focus. The question should help clarify whether the researcher is interested "
            "in experimental conditions, theoretical models, or computational parameters."
        )
        
        user_content = (
            f"Initial research query: '{research_query}'\n\n"
            f"Key topics identified in the literature: {', '.join(topics[:7])}\n\n"
            "Generate one clear follow-up question to help refine the research focus."
        )
        
        return self.query(user_content, system_content)