"""
Chemistry-specific agents for the Multi-Agent DFT Research System.
"""

import logging
from .base import Agent
from ..api.publication import PublicationAPI
from datetime import datetime

logger = logging.getLogger(__name__)


class ExperimentalChemistAgent(Agent):
    """Agent specialized in experimental chemistry."""
    
    def __init__(self, name="ExperimentalChemistAgent", model=None, config=None):
        """
        Initialize the experimental chemist agent.
        """
        super().__init__(name, model, config)
        self.config = config or {}
        
        # Initialize PublicationAPI with config
        from multi_agent_dft.api.publication import PublicationAPI
        self.pub_api = PublicationAPI(config=self.config)
        
        # Get agent-specific configuration
        agent_config = self.config.get('agents', {}).get('experimental_chemist', {})
        # Updated focus keywords
        self.focus_keywords = agent_config.get('focus_keywords', [
            "experiment", "synthesis", "characterization", "laboratory", "preparation",
            "measurement", "spectroscopy", "microscopy", "analysis", "purification",
            "crystallization", "reaction", "protocol", "procedure", "method",
            "technique", "instrument", "apparatus", "sample", "material"
        ])
    def _extract_key_terms(self, query):
        """Extract key terms from the user query"""
        # Simple extraction of important nouns and technical terms
        import re
        
        # Extract words that might be important (3+ letter words)
        words = re.findall(r'\b[a-zA-Z]{3,}\b', query.lower())
        
        # Basic list of stopwords to filter out
        stopwords = ["the", "and", "or", "in", "on", "at", "to", "for", "with", "by", 
                    "as", "from", "that", "this", "these", "those", "some", "such", 
                    "would", "could", "should", "will", "shall", "may", "might", "can"]
        
        # Get non-stopwords
        key_terms = [w for w in words if w not in stopwords]
        
        # Return at most 5 terms
        return key_terms[:5]
    def _create_targeted_query(self, user_query):
        """Create a more targeted query based on system detection"""
        query_lower = user_query.lower()
        
        # Detect system types
        if any(term in query_lower for term in ["crystal", "polymorph", "lattice"]):
            if self.__class__.__name__ == "TheoreticalChemistAgent":
                return "crystal structure prediction computational method"
            else:
                return "crystal characterization experimental technique"
        elif any(term in query_lower for term in ["metal", "catalyst", "oxide", "alloy"]):
            if self.__class__.__name__ == "TheoreticalChemistAgent":
                return "metal catalyst computational modeling"
            else:
                return "metal catalyst synthesis characterization"
        elif any(term in query_lower for term in ["organic", "molecule", "drug", "pharmaceutical"]):
            if self.__class__.__name__ == "TheoreticalChemistAgent":
                return "organic molecule computational chemistry"
            else:
                return "organic synthesis characterization method"
        elif any(term in query_lower for term in ["polymer", "macromolecule", "plastic"]):
            if self.__class__.__name__ == "TheoreticalChemistAgent":
                return "polymer modeling simulation"
            else:
                return "polymer synthesis characterization"
        else:
            # Default general query
            if self.__class__.__name__ == "TheoreticalChemistAgent":
                return "computational chemistry methods techniques"
            else:
                return "experimental chemistry methods techniques"
    
    def _rank_publications_by_relevance(self, publications, query):
        """Rank publications by relevance to the query"""
        from datetime import datetime
        
        # Get keywords from the query
        keywords = self._extract_key_terms(query)
        
        # Score each publication based on keyword presence in title and abstract
        scored_pubs = []
        for pub in publications:
            score = 0
            title = pub.get('title', '').lower()
            abstract = pub.get('abstract', '').lower()
            
            # Check title for keywords (higher weight)
            for keyword in keywords:
                if keyword in title:
                    score += 3
                if keyword in abstract:
                    score += 1
            
            # Bonus for recency
            try:
                year = int(pub.get('year', 0))
                current_year = datetime.now().year
                if year > current_year - 3:  # Published in the last 3 years
                    score += 2
                elif year > current_year - 7:  # Published in the last 7 years
                    score += 1
            except (ValueError, TypeError):
                pass
            
            # Bonus for review articles (often more comprehensive)
            if "review" in title.lower() or "advances" in title.lower() or "progress" in title.lower():
                score += 2
            
            # Store both score and publication
            scored_pubs.append((score, pub))
        
        # Sort by score (descending)
        scored_pubs.sort(key=lambda x: x[0], reverse=True)  # Use this corrected line
        
        # Return just the publications
        return [pub for _, pub in scored_pubs]


    def _generate_enhanced_summary(self, user_query, additional_context, report):
        """Generate an enhanced summary from the publication report"""
        if self.__class__.__name__ == "TheoreticalChemistAgent":
            system_content = (
                f"You are {self.name}, an expert theoretical chemist. "
                "Analyze the following publication report related to the user's query. "
                "Focus on providing clear, factual information about:"
                "\n1. Computational methods and theoretical approaches relevant to the query"
                "\n2. How computational chemistry has been applied to similar systems"
                "\n3. Key parameters, models, and algorithms reported in the literature"
                "\n4. Limitations and accuracy of theoretical predictions"
                "\n5. Recent advances in computational techniques for this area"
                "\nProvide specific examples from the literature when available."
                "\nIMPORTANT: Only include information that is explicitly mentioned in the publications. "
                "If specific information is not available, clearly state what is missing rather than fabricating details."
            )
        else:  # ExperimentalChemistAgent
            system_content = (
                f"You are {self.name}, an expert experimental chemist. "
                "Analyze the following publication report related to the user's query. "
                "Focus on providing clear, factual information about:"
                "\n1. Experimental techniques and methodologies relevant to the query"
                "\n2. Synthesis procedures and characterization methods"
                "\n3. Experimental conditions and parameters reported in the literature"
                "\n4. Challenges and solutions in experimental work"
                "\n5. Recent advances in experimental approaches for this area"
                "\nProvide specific examples from the literature when available."
                "\nIMPORTANT: Only include information that is explicitly mentioned in the publications. "
                "If specific information is not available, clearly state what is missing rather than fabricating details."
            )
            
        user_content = (
            f"Research Query: {user_query}\n\n"
            f"Additional Context: {additional_context}\n\n"
            f"Publication Report:\n{report}\n\n"
            f"Based ONLY on the information in the Publication Report, provide a detailed, factual summary "
            f"addressing the key points outlined above. Do NOT include any speculative information or details "
            f"that are not explicitly stated in the publications. If the publications do not contain specific information "
            f"on certain aspects, clearly indicate what information is missing rather than making it up."
        )
        
        # Get agent's summary
        return self.query(user_content, system_content)
    def _get_detailed_publication_data(self, publications, max_detailed=5):
        """Get detailed data from top publications including full-text when available"""
        # Sort by relevance/score if not already sorted
        if not publications:
            return []
        
        detailed_pubs = []
        for pub in publications[:max_detailed]:
            # Try to get full text
            full_text = self.pub_api.get_full_text(pub)
            if full_text:
                pub['full_text'] = full_text
                
            detailed_pubs.append(pub)
            
        return detailed_pubs
    def get_detailed_information(self, specific_topic, max_results=10):
        """
        Get more detailed information on a specific topic related to the research.
        
        Args:
            specific_topic (str): Specific aspect to research in more detail
            max_results (int): Maximum number of publications to analyze
            
        Returns:
            str: Detailed information on the specific topic
        """
        # Create a targeted query for the specific aspect
        query = f"{specific_topic} {' '.join(self.focus_keywords[:3])}"
        
        # Search for publications
        publications = self.pub_api.search(query, max_results=max_results)
        if not publications:
            return f"No detailed information found on {specific_topic}."
        
        # Analyze and rank publications
        ranked_publications = self._rank_publications_by_relevance(publications, specific_topic)
        top_publications = ranked_publications[:max_results]
        
        # Try to get full text for key publications
        detailed_publications = []
        for pub in top_publications[:3]:  # Get details for top 3
            try:
                full_text = self.pub_api.get_full_text(pub)
                if full_text:
                    pub['full_text'] = full_text
            except Exception as e:
                logger.warning(f"Error getting full text: {e}")
            detailed_publications.append(pub)
        
        # Generate focused report
        report = []
        report.append(f"# Detailed Information on: {specific_topic}")
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total publications analyzed: {len(top_publications)}")
        report.append("")
        
        # Report on publications
        report.append("## Key Publications")
        for i, pub in enumerate(top_publications[:5], 1):  # Top 5 publications
            report.append(f"### {i}. {pub.get('title', 'Untitled')}")
            report.append(f"**Authors**: {', '.join(pub.get('authors', ['Unknown']))}")
            report.append(f"**Source**: {pub.get('journal', 'Unknown')}, {pub.get('year', 'Unknown')}")
            report.append(f"**URL**: {pub.get('url', 'N/A')}")
            report.append("")
            report.append(f"**Abstract**: {pub.get('abstract', 'No abstract available')}")
            report.append("")
            
            # Add excerpts from full text if available
            if 'full_text' in pub:
                import re
                # Extract relevant sentences containing key terms related to the specific topic
                topic_terms = specific_topic.lower().split()
                sentences = re.split(r'(?<=[.!?])\s+', pub['full_text'])
                relevant_sentences = []
                
                for sentence in sentences:
                    if any(term in sentence.lower() for term in topic_terms):
                        relevant_sentences.append(sentence)
                
                if relevant_sentences:
                    report.append("**Key Excerpts**:")
                    # Limit to 3 sentences to keep it manageable
                    for sentence in relevant_sentences[:3]:
                        report.append(f"- {sentence}")
                    report.append("")
        
        report_text = "\n".join(report)
        
        # Generate a detailed summary
        system_content = (
            f"You are {self.name}, an expert experimental chemist. "
            f"Based on the following report about {specific_topic}, provide a detailed summary focused on "
            "experimental techniques and methodologies. Include specific procedures, conditions, and results "
            "mentioned in the publications. Only include factual information from the report."
        )
        
        user_content = f"Detailed Report on {specific_topic}:\n\n{report_text}\n\nProvide a factual summary based only on this information."
        
        return self.query(user_content, system_content)

    def summarize_with_references(self, user_query, additional_context="", max_results=15):
        """Enhanced summarize method that returns both summary and references."""
        
        # Extract key terms from the query
        key_terms = self._extract_key_terms(user_query)
        
        # Create search variations based on the agent type
        if self.__class__.__name__ == "TheoreticalChemistAgent":
            search_queries = [
                f"{' '.join(key_terms)} computational chemistry",
                f"{' '.join(key_terms)} DFT calculation",
                f"{' '.join(key_terms)} theoretical prediction",
                self._create_targeted_query(user_query)
            ]
        else:  # ExperimentalChemistAgent
            search_queries = [
                f"{' '.join(key_terms)} experimental characterization",
                f"{' '.join(key_terms)} synthesis method",
                f"{' '.join(key_terms)} laboratory technique",
                self._create_targeted_query(user_query)
            ]
        
        # Search for publications using multiple queries
        all_publications = []
        unique_ids = set()
        
        for query in search_queries:
            publications = self.pub_api.search(query, max_results=max_results//len(search_queries))
            # Deduplicate publications
            for pub in publications:
                pub_id = pub.get('id', pub.get('title', ''))
                if pub_id not in unique_ids:
                    unique_ids.add(pub_id)
                    all_publications.append(pub)
        
        if not all_publications:
            return {
                'summary': "No publications found related to the query.",
                'references': []
            }
        
        # Rank publications by relevance to the query
        ranked_publications = self._rank_publications_by_relevance(all_publications, user_query)
        
        # Take the top publications
        top_publications = ranked_publications[:max_results]
        
        # Analyze publications with focus keywords
        analysis = self.pub_api.analyze_publications(top_publications, self.focus_keywords)
        
        # Generate publication report
        report = self.pub_api.generate_report(top_publications, analysis)
        
        # Generate the summary with enhanced prompting
        summary = self._generate_enhanced_summary(user_query, additional_context, report)
        
        # Return both summary and references
        return {
            'summary': summary,
            'references': top_publications[:10]  # Return top 10 references used
        }


    def get_system_prompt(self):
        """
        Get the system prompt for the experimental chemist agent.
        
        Returns:
            str: Specialized system prompt for experimental chemistry.
        """
        return (
            f"You are {self.name}, an expert experimental chemist. "
            "Your expertise includes laboratory techniques, synthesis procedures, characterization methods, "
            "and experimental design. Analyze the provided information from a practical experimental perspective, "
            "focusing on synthesis methods, experimental conditions, characterization techniques, and reproducibility. "
            "Provide insights on experimental challenges, optimization strategies, and potential applications."
            "Your responses must be based solely on information retrieved from the information."
            " If the information doesn't have the answer, say 'I don't have information on that."
        )   

    def _detect_chemical_system(self, query_lower):
        """Detect chemical system type"""
        if any(term in query_lower for term in ["organic", "molecule", "conformer"]):
            return "organic"
        elif any(term in query_lower for term in ["crystal", "solid", "periodic"]):
            return "inorganic"  
        elif any(term in query_lower for term in ["catalyst", "metal", "surface"]):
            return "catalytic"
        else:
            return "molecular"
    def _detect_target_properties(self, query_lower):
        """Detect target properties"""
        properties = []
        if any(term in query_lower for term in ["energy", "stability", "conformer"]):
            properties.append("energetic")
        if any(term in query_lower for term in ["vibrational", "frequency", "infrared"]):
            properties.append("vibrational")
        if any(term in query_lower for term in ["thermodynamic", "entropy", "heat capacity"]):
            properties.append("thermodynamic")
        return properties

    def summarize(self, user_query, additional_context="", max_results=15):
        """Enhanced summarize method with intelligent search"""
        
        # Detect chemical system and target properties
        system_type = self._detect_chemical_system(user_query.lower())
        target_properties = self._detect_target_properties(user_query.lower())
        
        # Create intelligent search queries
        key_terms = self._extract_key_terms(user_query)
        
        if self.__class__.__name__ == "TheoreticalChemistAgent":
            search_queries = [
                f"{' '.join(key_terms)} DFT calculation",
                f"{' '.join(key_terms)} quantum chemistry",
                f"{' '.join(key_terms)} computational method",
                f"{system_type} computational chemistry {' '.join(key_terms[:2])}"
            ]
        else:  # ExperimentalChemistAgent
            search_queries = [
                f"{' '.join(key_terms)} experimental synthesis",
                f"{' '.join(key_terms)} characterization method",
                f"{' '.join(key_terms)} laboratory technique",
                f"{system_type} experimental {' '.join(key_terms[:2])}"
            ]
        
        # Search and process as before...
        all_publications = []
        for query in search_queries:
            publications = self.pub_api.search(query, max_results=max_results//len(search_queries))
            all_publications.extend(publications)
        
        # Remove duplicates and rank by relevance
        unique_pubs = []
        seen_ids = set()
        for pub in all_publications:
            pub_id = pub.get('id', pub.get('title', ''))
            if pub_id not in seen_ids:
                seen_ids.add(pub_id)
                unique_pubs.append(pub)
        
        ranked_pubs = self._rank_publications_by_relevance(unique_pubs, user_query)
        top_pubs = ranked_pubs[:max_results]
        
        # Generate report and summary as before
        analysis = self.pub_api.analyze_publications(top_pubs, self.focus_keywords)
        report = self.pub_api.generate_report(top_pubs, analysis)
        
        return self._generate_enhanced_summary(user_query, additional_context, report)



class TheoreticalChemistAgent(Agent):
    """Agent specialized in theoretical chemistry."""
    
    def __init__(self, name="TheoreticalChemistAgent", model=None, config=None):
        """
        Initialize the theoretical chemist agent.
        
        Args:
            name (str, optional): Agent name.
            model (str, optional): OpenAI model to use.
            config (dict, optional): Configuration dictionary.
        """
        super().__init__(name, model, config)
        self.config = config or {}
        
        # Initialize PublicationAPI with config
        from multi_agent_dft.api.publication import PublicationAPI
        self.pub_api = PublicationAPI(config=self.config)
        
        # Get agent-specific configuration
        agent_config = self.config.get('agents', {}).get('theoretical_chemist', {})
        self.focus_keywords = agent_config.get('focus_keywords', [
            "theory", "simulation", "computation", "calculation", "model", 
            "algorithm", "DFT", "quantum", "molecular dynamics", "ab initio",
            "functional", "basis set", "approximation", "accuracy", "method", 
            "prediction", "parameter", "energy", "property", "structure"
        ])    
    def get_system_prompt(self):
        """
        Get the system prompt for the theoretical chemist agent.
        
        Returns:
            str: Specialized system prompt for theoretical chemistry.
        """
        return (
            f"You are {self.name}, an expert theoretical chemist. "
            "Your expertise includes quantum mechanics, computational chemistry, molecular modeling, "
            "density functional theory, and theoretical frameworks. Analyze the provided information "
            "from a theoretical perspective, focusing on models, computational methods, algorithms, "
            "and theoretical predictions. Provide insights on theoretical challenges, methodological improvements, "
            "and connections between theory and experiment."
        )
    def _extract_key_terms(self, query):
        """Extract key terms from the user query"""
        # Simple extraction of important nouns and technical terms
        import re
        
        # Extract words that might be important (3+ letter words)
        words = re.findall(r'\b[a-zA-Z]{3,}\b', query.lower())
        
        # Basic list of stopwords to filter out
        stopwords = ["the", "and", "or", "in", "on", "at", "to", "for", "with", "by", 
                    "as", "from", "that", "this", "these", "those", "some", "such", 
                    "would", "could", "should", "will", "shall", "may", "might", "can"]
        
        # Get non-stopwords
        key_terms = [w for w in words if w not in stopwords]
        
        # Return at most 5 terms
        return key_terms[:5]
    

    def summarize_with_references(self, user_query, additional_context="", max_results=15):
        """Enhanced summarize method that returns both summary and references."""
        
        # Extract key terms from the query
        key_terms = self._extract_key_terms(user_query)
        
        # Create search variations based on the agent type
        if self.__class__.__name__ == "TheoreticalChemistAgent":
            search_queries = [
                f"{' '.join(key_terms)} computational chemistry",
                f"{' '.join(key_terms)} DFT calculation",
                f"{' '.join(key_terms)} theoretical prediction",
                self._create_targeted_query(user_query)
            ]
        else:  # ExperimentalChemistAgent
            search_queries = [
                f"{' '.join(key_terms)} experimental characterization",
                f"{' '.join(key_terms)} synthesis method",
                f"{' '.join(key_terms)} laboratory technique",
                self._create_targeted_query(user_query)
            ]
        
        # Search for publications using multiple queries
        all_publications = []
        unique_ids = set()
        
        for query in search_queries:
            publications = self.pub_api.search(query, max_results=max_results//len(search_queries))
            # Deduplicate publications
            for pub in publications:
                pub_id = pub.get('id', pub.get('title', ''))
                if pub_id not in unique_ids:
                    unique_ids.add(pub_id)
                    all_publications.append(pub)
        
        if not all_publications:
            return {
                'summary': "No publications found related to the query.",
                'references': []
            }
        
        # Rank publications by relevance to the query
        ranked_publications = self._rank_publications_by_relevance(all_publications, user_query)
        
        # Take the top publications
        top_publications = ranked_publications[:max_results]
        
        # Analyze publications with focus keywords
        analysis = self.pub_api.analyze_publications(top_publications, self.focus_keywords)
        
        # Generate publication report
        report = self.pub_api.generate_report(top_publications, analysis)
        
        # Generate the summary with enhanced prompting
        summary = self._generate_enhanced_summary(user_query, additional_context, report)
        
        # Return both summary and references
        return {
            'summary': summary,
            'references': top_publications[:10]  # Return top 10 references used
        }



    def _create_targeted_query(self, user_query):
        """Create a more targeted query based on system detection"""
        query_lower = user_query.lower()
        
        # Detect system types
        if any(term in query_lower for term in ["crystal", "polymorph", "lattice"]):
            if self.__class__.__name__ == "TheoreticalChemistAgent":
                return "crystal structure prediction computational method"
            else:
                return "crystal characterization experimental technique"
        elif any(term in query_lower for term in ["metal", "catalyst", "oxide", "alloy"]):
            if self.__class__.__name__ == "TheoreticalChemistAgent":
                return "metal catalyst computational modeling"
            else:
                return "metal catalyst synthesis characterization"
        elif any(term in query_lower for term in ["organic", "molecule", "drug", "pharmaceutical"]):
            if self.__class__.__name__ == "TheoreticalChemistAgent":
                return "organic molecule computational chemistry"
            else:
                return "organic synthesis characterization method"
        elif any(term in query_lower for term in ["polymer", "macromolecule", "plastic"]):
            if self.__class__.__name__ == "TheoreticalChemistAgent":
                return "polymer modeling simulation"
            else:
                return "polymer synthesis characterization"
        else:
            # Default general query
            if self.__class__.__name__ == "TheoreticalChemistAgent":
                return "computational chemistry methods techniques"
            else:
                return "experimental chemistry methods techniques"
    def _rank_publications_by_relevance(self, publications, query):
        """Rank publications by relevance to the query"""
        from datetime import datetime
        
        # Get keywords from the query
        keywords = self._extract_key_terms(query)
        
        # Score each publication based on keyword presence in title and abstract
        scored_pubs = []
        for pub in publications:
            score = 0
            title = pub.get('title', '').lower()
            abstract = pub.get('abstract', '').lower()
            
            # Check title for keywords (higher weight)
            for keyword in keywords:
                if keyword in title:
                    score += 3
                if keyword in abstract:
                    score += 1
            
            # Bonus for recency
            try:
                year = int(pub.get('year', 0))
                current_year = datetime.now().year
                if year > current_year - 3:  # Published in the last 3 years
                    score += 2
                elif year > current_year - 7:  # Published in the last 7 years
                    score += 1
            except (ValueError, TypeError):
                pass
            
            # Bonus for review articles (often more comprehensive)
            if "review" in title.lower() or "advances" in title.lower() or "progress" in title.lower():
                score += 2
            
            # Store both score and publication
            scored_pubs.append((score, pub))
        
        # Sort by score (descending)
        scored_pubs.sort(key=lambda x: x[0], reverse=True)  # Use this corrected line
        
        # Return just the publications
        return [pub for _, pub in scored_pubs]

    def _generate_enhanced_summary(self, user_query, additional_context, report):
        """Generate an enhanced summary from the publication report with citations"""
        if self.__class__.__name__ == "TheoreticalChemistAgent":
            system_content = (
                f"You are {self.name}, an expert theoretical chemist. "
                "Analyze the following publication report related to the user's query. "
                "Focus on providing clear, factual information about:"
                "\n1. Computational methods and theoretical approaches relevant to the query"
                "\n2. How computational chemistry has been applied to similar systems"
                "\n3. Key parameters, models, and algorithms reported in the literature"
                "\n4. Limitations and accuracy of theoretical predictions"
                "\n5. Recent advances in computational techniques for this area"
                "\nProvide specific examples from the literature when available."
                "\nIMPORTANT: Only include information that is explicitly mentioned in the publications. "
                "If specific information is not available, clearly state what is missing rather than fabricating details."
                "\nIMPORTANT: Whenever you cite information from a specific publication, include the citation number in square brackets [X] as it appears in the report. "
                "For example: 'Hybrid functionals have been shown to provide more accurate energy rankings for molecular crystals [3].' "
                "The references are listed at the end of the report."
            )
        else:  # ExperimentalChemistAgent
            system_content = (
                f"You are {self.name}, an expert experimental chemist. "
                "Analyze the following publication report related to the user's query. "
                "Focus on providing clear, factual information about:"
                "\n1. Experimental techniques and methodologies relevant to the query"
                "\n2. Synthesis procedures and characterization methods"
                "\n3. Experimental conditions and parameters reported in the literature"
                "\n4. Challenges and solutions in experimental work"
                "\n5. Recent advances in experimental approaches for this area"
                "\nProvide specific examples from the literature when available."
                "\nIMPORTANT: Only include information that is explicitly mentioned in the publications. "
                "If specific information is not available, clearly state what is missing rather than fabricating details."
                "\nIMPORTANT: Whenever you cite information from a specific publication, include the citation number in square brackets [X] as it appears in the report. "
                "For example: 'Single crystal X-ray diffraction has been used to characterize polymorphs of small organic molecules [2].' "
                "The references are listed at the end of the report."
            )
            
        user_content = (
            f"Research Query: {user_query}\n\n"
            f"Additional Context: {additional_context}\n\n"
            f"Publication Report:\n{report}\n\n"
            f"Based ONLY on the information in the Publication Report, provide a detailed, factual summary "
            f"addressing the key points outlined above. Use citation numbers [X] when referring to specific publications. "
            f"Make sure to properly cite all information from the report with the appropriate citation numbers. "
            f"Do NOT include any speculative information or details that are not explicitly stated in the publications. "
            f"If the publications do not contain specific information on certain aspects, clearly indicate what information is missing."
        )
        
        # Get agent's summary
        return self.query(user_content, system_content)

    def _get_detailed_publication_data(self, publications, max_detailed=5):
        """Get detailed data from top publications including full-text when available"""
        # Sort by relevance/score if not already sorted
        if not publications:
            return []
        
        detailed_pubs = []
        for pub in publications[:max_detailed]:
            # Try to get full text
            full_text = self.pub_api.get_full_text(pub)
            if full_text:
                pub['full_text'] = full_text
            
            # Get any DFT parameters if this is theoretical research
            if self.__class__.__name__ == "TheoreticalChemistAgent":
                try:
                    dft_parameters = self.pub_api.extract_parameters(pub, "dft", "crystal")
                    if dft_parameters:
                        pub['dft_parameters'] = dft_parameters
                except Exception as e:
                    logger.warning(f"Error extracting DFT parameters: {e}")
                    
            detailed_pubs.append(pub)
            
        return detailed_pubs
    def get_detailed_information(self, specific_topic, max_results=10):
        """
        Get more detailed information on a specific topic related to the research.
        
        Args:
            specific_topic (str): Specific aspect to research in more detail
            max_results (int): Maximum number of publications to analyze
            
        Returns:
            str: Detailed information on the specific topic
        """
        # Create a targeted query for the specific aspect
        query = f"{specific_topic} {' '.join(self.focus_keywords[:3])}"
        
        # Search for publications
        publications = self.pub_api.search(query, max_results=max_results)
        if not publications:
            return f"No detailed information found on {specific_topic}."
        
        # Analyze and rank publications
        ranked_publications = self._rank_publications_by_relevance(publications, specific_topic)
        top_publications = ranked_publications[:max_results]
        
        # Try to get full text for key publications
        detailed_publications = []
        for pub in top_publications[:3]:  # Get details for top 3
            try:
                full_text = self.pub_api.get_full_text(pub)
                if full_text:
                    pub['full_text'] = full_text
            except Exception as e:
                logger.warning(f"Error getting full text: {e}")
            detailed_publications.append(pub)
        
        # Generate focused report
        report = []
        report.append(f"# Detailed Information on: {specific_topic}")
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total publications analyzed: {len(top_publications)}")
        report.append("")
        
        # Report on publications
        report.append("## Key Publications")
        for i, pub in enumerate(top_publications[:5], 1):  # Top 5 publications
            report.append(f"### {i}. {pub.get('title', 'Untitled')}")
            report.append(f"**Authors**: {', '.join(pub.get('authors', ['Unknown']))}")
            report.append(f"**Source**: {pub.get('journal', 'Unknown')}, {pub.get('year', 'Unknown')}")
            report.append(f"**URL**: {pub.get('url', 'N/A')}")
            report.append("")
            report.append(f"**Abstract**: {pub.get('abstract', 'No abstract available')}")
            report.append("")
            
            # Add excerpts from full text if available
            if 'full_text' in pub:
                import re
                # Extract relevant sentences containing key terms related to the specific topic
                topic_terms = specific_topic.lower().split()
                sentences = re.split(r'(?<=[.!?])\s+', pub['full_text'])
                relevant_sentences = []
                
                for sentence in sentences:
                    if any(term in sentence.lower() for term in topic_terms):
                        relevant_sentences.append(sentence)
                
                if relevant_sentences:
                    report.append("**Key Excerpts**:")
                    # Limit to 3 sentences to keep it manageable
                    for sentence in relevant_sentences[:3]:
                        report.append(f"- {sentence}")
                    report.append("")
        
        report_text = "\n".join(report)
        
        # Generate a detailed summary
        if self.__class__.__name__ == "TheoreticalChemistAgent":
            system_content = (
                f"You are {self.name}, an expert theoretical chemist. "
                f"Based on the following report about {specific_topic}, provide a detailed summary focused on "
                "theoretical and computational aspects. Include specific methods, parameters, and findings "
                "mentioned in the publications. Only include factual information from the report."
            )
        else:
            system_content = (
                f"You are {self.name}, an expert experimental chemist. "
                f"Based on the following report about {specific_topic}, provide a detailed summary focused on "
                "experimental techniques and methodologies. Include specific procedures, conditions, and results "
                "mentioned in the publications. Only include factual information from the report."
            )
        
        user_content = f"Detailed Report on {specific_topic}:\n\n{report_text}\n\nProvide a factual summary based only on this information."
        
        return self.query(user_content, system_content)   
    def summarize(self, user_query, additional_context="", max_results=15):
        """
        Enhanced summarize method with better search strategies while remaining general
        """
        # Extract key terms from the query
        key_terms = self._extract_key_terms(user_query)
        
        # Create search variations based on the agent type, but keep them general
        if self.__class__.__name__ == "TheoreticalChemistAgent":
            # For theoretical chemist, create variations focused on computational aspects
            search_queries = [
                f"{' '.join(key_terms)} computational chemistry",
                f"{' '.join(key_terms)} DFT calculation",
                f"{' '.join(key_terms)} theoretical prediction",
                # Add a more specific query based on detected system type
                self._create_targeted_query(user_query)
            ]
        else:  # ExperimentalChemistAgent
            # For experimental chemist, create variations focused on experimental aspects
            search_queries = [
                f"{' '.join(key_terms)} experimental characterization",
                f"{' '.join(key_terms)} synthesis method",
                f"{' '.join(key_terms)} laboratory technique",
                # Add a more specific query based on detected system type
                self._create_targeted_query(user_query)
            ]
        
        # Search for publications using multiple queries
        all_publications = []
        unique_ids = set()
        
        for query in search_queries:
            publications = self.pub_api.search(query, max_results=max_results//len(search_queries))
            # Deduplicate publications
            for pub in publications:
                pub_id = pub.get('id', '')
                if pub_id not in unique_ids:
                    unique_ids.add(pub_id)
                    all_publications.append(pub)
        
        if not all_publications:
            return "No publications found related to the query."
        
        # Rank publications by relevance to the query
        ranked_publications = self._rank_publications_by_relevance(all_publications, user_query)
        
        # Take the top publications
        top_publications = ranked_publications[:max_results]
        
        # Analyze publications with focus keywords
        analysis = self.pub_api.analyze_publications(top_publications, self.focus_keywords)
        
        # Generate publication report
        report = self.pub_api.generate_report(top_publications, analysis)
        
        # Generate the summary with enhanced prompting
        return self._generate_enhanced_summary(user_query, additional_context, report)

