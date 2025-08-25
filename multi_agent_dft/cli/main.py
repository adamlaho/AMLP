def run_interactive_session(config):
    """
    Run an interactive session.
    
    Args:
        config (dict): Configuration dictionary.
    
    Returns:
        int: Exit code.
    """
    from ..dft.cp2k import interactive_cp2k_config
    from ..dft.vasp import interactive_vasp_config
    from ..dft.gaussian import interactive_gaussian_config
    
    print("\nWelcome to the Multi-Agent DFT Research System Interactive Session\n")
    
    try:
        # Initialize agents
        print("Initializing agents...")
        exp_agent = ExperimentalChemistAgent(config=config)
        theo_agent = TheoreticalChemistAgent(config=config)
        sup1_agent = SupervisorAgent("Integration", config=config)
        gaussian_expert = GaussianExpertAgent(config=config)
        vasp_expert = VASPExpertAgent(config=config)
        cp2k_expert = CP2KExpertAgent(config=config)
        sup2_agent = SupervisorAgent("DFT_Recommendation", config=config)
        
        while True:
            print("\n" + "="*50)
            print("Available commands:")
            print("1. Research")
            print("2. Process structure files directly")
            print("3. Generate DFT input")
            print("4. Publication search")
            print("0. Exit")
            
            choice = input("\nEnter your choice (0-4): ")
            
            if choice == "0":
                print("\nGoodbye!")
                return 0
            
            elif choice == "1":
                # Research mode
                query = input("\nEnter your research topic or question: ")
                max_results = int(input("Maximum number of publications to analyze (default 10): ") or "10")
                
                followup_question = sup1_agent.generate_followup_question(query)
                answer = input(f"\n{followup_question}\nYour answer (press Enter to skip): ")
                additional_context = f"{followup_question} Answer: {answer}\n" if answer else ""
                
                # Refine the query
                refined_query = f"{query} with context: {additional_context}" if additional_context else query
                print(f"\nRefined research query:\n{refined_query}\n")
                
                # Get experimental and theoretical summaries
                print("Getting experimental summary...")
                exp_summary = exp_agent.summarize(query, additional_context, max_results)
                print("Getting theoretical summary...")
                theo_summary = theo_agent.summarize(query, additional_context, max_results)
                
                print("\n--- Experimental Chemist Summary ---")
                print(exp_summary)
                print("\n--- Theoretical Chemist Summary ---")
                print(theo_summary)
                
                # Integrate experimental and theoretical summaries
                print("Integrating summaries...")
                integrated_content = f"Experimental Summary:\n{exp_summary}\n\nTheoretical Summary:\n{theo_summary}"
                sup1_report = sup1_agent.integrate(integrated_content)
                
                print("\n--- Supervisor 1 Integrated Report ---")
                print(sup1_report)
                
                # Get DFT expert reports
                print("\nEngaging DFT Expert Agents...\n")
                
                print("Getting GAUSSIAN expert report...")
                gaussian_report = gaussian_expert.analyze(refined_query)
                print("Getting VASP expert report...")
                vasp_report = vasp_expert.analyze(refined_query)
                print("Getting CP2K expert report...")
                cp2k_report = cp2k_expert.analyze(refined_query)
                
                print("\n--- GAUSSIAN Expert Report ---")
                print(gaussian_report)
                print("\n--- VASP Expert Report ---")
                print(vasp_report)
                print("\n--- CP2K Expert Report ---")
                print(cp2k_report)
                
                # Get final DFT recommendation
                print("Getting DFT recommendation...")
                dft_content = f"GAUSSIAN Report:\n{gaussian_report}\n\nVASP Report:\n{vasp_report}\n\nCP2K Report:\n{cp2k_report}"
                sup2_report = sup2_agent.integrate(dft_content)
                
                print("\n--- Supervisor 2 Final Recommendation ---")
                print(sup2_report)
                
                # Ask if the user wants to save the reports
                save_reports = input("\nDo you want to save the reports? (y/n): ").lower() == 'y'
                if save_reports:
                    output_dir = Path(input("Enter output directory: ") or "./research_output")
                    output_dir.mkdir(parents=True, exist_ok=True)
                    
                    with open(output_dir / "experimental_summary.md", "w") as f:
                        f.write(f"# Experimental Summary for: {query}\n\n{exp_summary}")
                    
                    with open(output_dir / "theoretical_summary.md", "w") as f:
                        f.write(f"# Theoretical Summary for: {query}\n\n{theo_summary}")
                    
                    with open(output_dir / "integrated_report.md", "w") as f:
                        f.write(f"# Integrated Report for: {query}\n\n{sup1_report}")
                    
                    with open(output_dir / "dft_recommendation.md", "w") as f:
                        f.write(f"# DFT Recommendation for: {query}\n\n{sup2_report}")
                    
                    # Save individual DFT reports
                    dft_dir = output_dir / "dft_reports"
                    dft_dir.mkdir(exist_ok=True)
                    
                    with open(dft_dir / "gaussian_report.md", "w") as f:
                        f.write(f"# GAUSSIAN Expert Report for: {query}\n\n{gaussian_report}")
                    
                    with open(dft_dir / "vasp_report.md", "w") as f:
                        f.write(f"# VASP Expert Report for: {query}\n\n{vasp_report}")
                    
                    with open(dft_dir / "cp2k_report.md", "w") as f:
                        f.write(f"# CP2K Expert Report for: {query}\n\n{cp2k_report}")
                    
                    print(f"\nReports saved to: {output_dir}")
                
                # Ask if the user wants to process structure files based on recommendations
                process_files = input("\nWould you like to process structure files based on these recommendations? (y/n): ").lower() == 'y'
                if process_files:
                    # Ask which DFT code to use
                    print("\nBased on the recommendations, which DFT code would you like to use?")
                    print("1. CP2K")
                    print("2. VASP")
                    print("3. Gaussian")
                    dft_choice = input("Enter your choice (1-3): ")
                    
                    if dft_choice == "1":
                        dft_code = "cp2k"
                        print("\nGenerating CP2K input configuration...")
                        dft_config = interactive_cp2k_config()
                    elif dft_choice == "2":
                        dft_code = "vasp"
                        print("\nGenerating VASP input configuration...")
                        dft_config, kpoints_config = interactive_vasp_config()
                    elif dft_choice == "3":
                        dft_code = "gaussian"
                        print("\nGenerating Gaussian input configuration...")
                        dft_config = interactive_gaussian_config()
                    else:
                        print("Invalid choice. Using CP2K as default.")
                        dft_code = "cp2k"
                        dft_config = interactive_cp2k_config()
                    
                    # Get structure file(s) path
                    file_path = input("\nEnter the path to your structure file or directory containing structure files: ")
                    file_path = Path(file_path).expanduser().resolve()
                    
                    if not file_path.exists():
                        print(f"Error: Path does not exist: {file_path}")
                        continue
                    
                    # Get output directory
                    if save_reports:
                        default_output = output_dir / "dft_inputs"
                    else:
                        default_output = "./dft_inputs"
                    
                    output_dir_path = input(f"Enter output directory for DFT input files (default: {default_output}): ") or default_output
                    output_dir_for_dft = Path(output_dir_path)
                    output_dir_for_dft.mkdir(parents=True, exist_ok=True)
                    
                    # Process the file(s)
                    if file_path.is_file():
                        # Process a single file
                        output_file = process_structure_file(
                            file_path,
                            output_dir_for_dft,
                            dft_code,
                            dft_config
                        )
                        
                        if output_file:
                            print(f"\nGenerated DFT input file: {output_file}")
                        else:
                            print(f"\nFailed to process file: {file_path}")
                    
                    elif file_path.is_dir():
                        # Process a directory
                        pattern = input("File pattern to match (default: *.{xyz,cif}): ") or "*.{xyz,cif}"
                        
                        output_files = process_structure_files(
                            file_path,
                            output_dir_for_dft,
                            pattern,
                            dft_code,
                            dft_config
                        )
                        
                        if output_files:
                            print(f"\nGenerated {len(output_files)} DFT input files")
                        else:
                            print(f"\nNo files processed in directory: {file_path}")
            
            elif choice == "2":
                # Process structure files directly
                print("\n--- Direct Structure File Processing ---")
                
                # Ask which DFT code to use
                print("Which DFT code would you like to use?")
                print("1. CP2K")
                print("2. VASP")
                print("3. Gaussian")
                dft_choice = input("Enter your choice (1-3): ")
                
                if dft_choice == "1":
                    dft_code = "cp2k"
                    print("\nGenerating CP2K input configuration...")
                    dft_config = interactive_cp2k_config()
                elif dft_choice == "2":
                    dft_code = "vasp"
                    print("\nGenerating VASP input configuration...")
                    dft_config, kpoints_config = interactive_vasp_config()
                elif dft_choice == "3":
                    dft_code = "gaussian"
                    print("\nGenerating Gaussian input configuration...")
                    dft_config = interactive_gaussian_config()
                else:
                    print("Invalid choice. Using CP2K as default.")
                    dft_code = "cp2k"
                    dft_config = interactive_cp2k_config()
                
                # Get structure file(s) path
                file_path = input("\nEnter the path to your structure file or directory containing structure files: ")
                file_path = Path(file_path).expanduser().resolve()
                
                if not file_path.exists():
                    print(f"Error: Path does not exist: {file_path}")
                    continue
                
                # Get output directory
                output_dir = Path(input("Enter output directory for DFT input files (default: ./dft_inputs): ") or "./dft_inputs")
                output_dir.mkdir(parents=True, exist_ok=True)
                
                if file_path.is_file():
                    # Process a single file
                    output_file = process_structure_file(
                        file_path,
                        output_dir,
                        dft_code,
                        dft_config
                    )
                    
                    if output_file:
                        print(f"\nGenerated DFT input file: {output_file}")
                    else:
                        print(f"\nFailed to process file: {file_path}")
                
                elif file_path.is_dir():
                    # Process a directory
                    pattern = input("File pattern to match (default: *.{xyz,cif}): ") or "*.{xyz,cif}"
                    
                    output_files = process_structure_files(
                        file_path,
                        output_dir,
                        pattern,
                        dft_code,
                        dft_config
                    )
                    
                    if output_files:
                        print(f"\nGenerated {len(output_files)} DFT input files")
                    else:
                        print(f"\nNo files processed in directory: {file_path}")
            
            elif choice == "3":
                # Generate DFT input interactively
                print("\n--- Generate DFT Input Files ---")
                
                dft_code = input("\nWhich DFT code would you like to use? (cp2k/vasp/gaussian) [default: cp2k]: ") or "cp2k"
                
                if dft_code.lower() == "cp2k":
                    print("\nGenerating CP2K input configuration...")
                    cp2k_config = interactive_cp2k_config()
                    
                    file_type = input("Do you have CIF files or XYZ files? (Enter CIF/XYZ): ")
                    file_dir = input("Enter the directory path containing your structure files: ")
                    out_dir = input("Enter the output directory path where input files will be generated: ")
                    
                    if file_type.lower() == "cif":
                        process_structure_files(file_dir, out_dir, "*.cif", "cp2k", cp2k_config)
                    elif file_type.lower() == "xyz":
                        process_structure_files(file_dir, out_dir, "*.xyz", "cp2k", cp2k_config)
                    else:
                        print("Invalid file type.")
                
                elif dft_code.lower() == "vasp":
                    print("\nGenerating VASP input configuration...")
                    incar_config, kpoints_config = interactive_vasp_config()
                    
                    file_dir = input("Enter the directory path containing your structure files: ")
                    out_dir = input("Enter the output directory path where input files will be generated: ")
                    
                    process_structure_files(file_dir, out_dir, "*.{xyz,cif}", "vasp", incar_config)
                
                elif dft_code.lower() == "gaussian":
                    print("\nGenerating Gaussian input configuration...")
                    gaussian_config = interactive_gaussian_config()
                    
                    file_dir = input("Enter the directory path containing your structure files: ")
                    out_dir = input("Enter the output directory path where input files will be generated: ")
                    
                    process_structure_files(file_dir, out_dir, "*.{xyz,cif}", "gaussian", gaussian_config)
                
                else:
                    print(f"Unsupported DFT code: {dft_code}")
            
            elif choice == "4":
                # Publication search
                pub_api = PublicationAPI(config=config)
                
                query = input("\nEnter your search query: ")
                max_results = int(input("Maximum number of results (default 10): ") or "10")
                
                print(f"\nSearching for publications related to: {query}")
                publications = pub_api.search(query, max_results=max_results)
                
                if not publications:
                    print("No publications found.")
                    continue
                
                print(f"\nFound {len(publications)} publications.")
                
                # Ask for keywords to analyze
                keywords = input("\nEnter keywords to analyze (comma-separated, press Enter to skip): ")
                if keywords:
                    keywords = [k.strip() for k in keywords.split(",")]
                    analysis = pub_api.analyze_publications(publications, keywords)
                else:
                    analysis = pub_api.analyze_publications(publications)
                
                # Generate report
                report = pub_api.generate_report(publications, analysis)
                print("\n" + "="*50)
                print(report)
                
                # Ask if the user wants to save the report
                save_report = input("\nDo you want to save the report? (y/n): ").lower() == 'y'
                if save_report:
                    output_dir = Path(input("Enter output directory: ") or "./publication_reports")
                    output_dir.mkdir(parents=True, exist_ok=True)
                    
                    report_file = output_dir / f"publication_report_{query.replace(' ', '_')}.md"
                    with open(report_file, "w") as f:
                        f.write(report)
                    
                    print(f"\nReport saved to: {report_file}")
            
            else:
                print("\nInvalid choice. Please try again.")
    
    except KeyboardInterrupt:
        print("\n\nSession terminated by user.")
        return 0
    
    except Exception as e:
        logger.error(f"Error in interactive session: {e}", exc_info=True)
        print(f"\nError: {e}")
        return 1"""
Command Line Interface for the Multi-Agent DFT Research System.
"""

import os
import sys
import logging
import argparse
from pathlib import Path

from ..utils.logging import setup_logging, get_default_log_file, log_system_info
from ..config import load_config, load_dft_config
from ..agents.chemistry_agents import ExperimentalChemistAgent, TheoreticalChemistAgent
from ..agents.dft_agents import GaussianExpertAgent, VASPExpertAgent, CP2KExpertAgent
from ..agents.supervisor import SupervisorAgent
from ..file_processing import process_structure_file, process_structure_files
from ..api.publication import PublicationAPI


logger = logging.getLogger(__name__)


def parse_args():
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Multi-Agent DFT Research System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # General options
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                       default="INFO", help="Logging level")
    parser.add_argument("--log-file", help="Path to log file")
    parser.add_argument("--no-console", action="store_true", help="Disable console logging")
    
    # Sub-commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Research command
    research_parser = subparsers.add_parser("research", help="Research a topic")
    research_parser.add_argument("query", help="Research query")
    research_parser.add_argument("--max-results", type=int, default=10,
                              help="Maximum number of publications to analyze")
    research_parser.add_argument("--output", help="Output directory for research reports")
    
    # Process command
    process_parser = subparsers.add_parser("process", help="Process structure files")
    process_parser.add_argument("input", help="Input file or directory")
    process_parser.add_argument("--output", help="Output directory")
    process_parser.add_argument("--dft", choices=["cp2k", "vasp", "gaussian"], default="cp2k",
                             help="DFT code to use")
    process_parser.add_argument("--recursive", action="store_true", help="Process directories recursively")
    process_parser.add_argument("--pattern", default="*.{xyz,cif}", help="File pattern to match")
    
    # Interactive command
    subparsers.add_parser("interactive", help="Start interactive session")
    
    # Version command
    subparsers.add_parser("version", help="Show version information")
    
    return parser.parse_args()


def run_research_command(args, config):
    """
    Run the research command.
    
    Args:
        args (argparse.Namespace): Command line arguments.
        config (dict): Configuration dictionary.
    
    Returns:
        int: Exit code.
    """
    logger.info(f"Running research command for query: {args.query}")
    
    # Create output directory if specified
    output_dir = None
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Initialize agents
        logger.info("Initializing agents...")
        exp_agent = ExperimentalChemistAgent(config=config)
        theo_agent = TheoreticalChemistAgent(config=config)
        sup1_agent = SupervisorAgent("Integration", config=config)
        gaussian_expert = GaussianExpertAgent(config=config)
        vasp_expert = VASPExpertAgent(config=config)
        cp2k_expert = CP2KExpertAgent(config=config)
        sup2_agent = SupervisorAgent("DFT_Recommendation", config=config)
        
        # Get initial follow-up question
        logger.info("Generating follow-up question...")
        followup_question = sup1_agent.generate_followup_question(args.query)
        print(f"\n{followup_question}")
        answer = input("Your answer (press Enter to skip): ")
        additional_context = f"{followup_question} Answer: {answer}\n" if answer else ""
        
        # Refine the query
        logger.info("Refining research query...")
        refined_query = f"{args.query} with context: {additional_context}" if additional_context else args.query
        print(f"\nRefined research query:\n{refined_query}\n")
        
        # Get experimental and theoretical summaries
        logger.info("Getting experimental summary...")
        exp_summary = exp_agent.summarize(args.query, additional_context, args.max_results)
        logger.info("Getting theoretical summary...")
        theo_summary = theo_agent.summarize(args.query, additional_context, args.max_results)
        
        print("\n--- Experimental Chemist Summary ---")
        print(exp_summary)
        print("\n--- Theoretical Chemist Summary ---")
        print(theo_summary)
        
        # Integrate experimental and theoretical summaries
        logger.info("Integrating summaries...")
        integrated_content = f"Experimental Summary:\n{exp_summary}\n\nTheoretical Summary:\n{theo_summary}"
        sup1_report = sup1_agent.integrate(integrated_content)
        
        print("\n--- Supervisor 1 Integrated Report ---")
        print(sup1_report)
        
        # Get DFT expert reports
        logger.info("Getting DFT expert reports...")
        print("\nEngaging DFT Expert Agents...\n")
        
        gaussian_report = gaussian_expert.analyze(refined_query)
        vasp_report = vasp_expert.analyze(refined_query)
        cp2k_report = cp2k_expert.analyze(refined_query)
        
        print("\n--- GAUSSIAN Expert Report ---")
        print(gaussian_report)
        print("\n--- VASP Expert Report ---")
        print(vasp_report)
        print("\n--- CP2K Expert Report ---")
        print(cp2k_report)
        
        # Get final DFT recommendation
        logger.info("Getting DFT recommendation...")
        dft_content = f"GAUSSIAN Report:\n{gaussian_report}\n\nVASP Report:\n{vasp_report}\n\nCP2K Report:\n{cp2k_report}"
        sup2_report = sup2_agent.integrate(dft_content)
        
        print("\n--- Supervisor 2 Final Recommendation ---")
        print(sup2_report)
        
        # Save reports if output directory is specified
        if output_dir:
            logger.info(f"Saving reports to {output_dir}")
            
            with open(output_dir / "experimental_summary.md", "w") as f:
                f.write(f"# Experimental Summary for: {args.query}\n\n{exp_summary}")
            
            with open(output_dir / "theoretical_summary.md", "w") as f:
                f.write(f"# Theoretical Summary for: {args.query}\n\n{theo_summary}")
            
            with open(output_dir / "integrated_report.md", "w") as f:
                f.write(f"# Integrated Report for: {args.query}\n\n{sup1_report}")
            
            with open(output_dir / "dft_recommendation.md", "w") as f:
                f.write(f"# DFT Recommendation for: {args.query}\n\n{sup2_report}")
            
            # Save individual DFT reports
            dft_dir = output_dir / "dft_reports"
            dft_dir.mkdir(exist_ok=True)
            
            with open(dft_dir / "gaussian_report.md", "w") as f:
                f.write(f"# GAUSSIAN Expert Report for: {args.query}\n\n{gaussian_report}")
            
            with open(dft_dir / "vasp_report.md", "w") as f:
                f.write(f"# VASP Expert Report for: {args.query}\n\n{vasp_report}")
            
            with open(dft_dir / "cp2k_report.md", "w") as f:
                f.write(f"# CP2K Expert Report for: {args.query}\n\n{cp2k_report}")
            
            print(f"\nReports saved to: {output_dir}")
        
        # Ask if the user wants to process structure files based on recommendations
        process_files = input("\nWould you like to process structure files based on these recommendations? (y/n): ").lower() == 'y'
        if process_files:
            # Ask which DFT code to use
            print("\nBased on the recommendations, which DFT code would you like to use?")
            print("1. CP2K")
            print("2. VASP")
            print("3. Gaussian")
            dft_choice = input("Enter your choice (1-3): ")
            
            if dft_choice == "1":
                dft_code = "cp2k"
            elif dft_choice == "2":
                dft_code = "vasp"
            elif dft_choice == "3":
                dft_code = "gaussian"
            else:
                print("Invalid choice. Using CP2K as default.")
                dft_code = "cp2k"
            
            # Get structure file(s) path
            file_path = input("\nEnter the path to your structure file or directory containing structure files: ")
            file_path = Path(file_path).expanduser().resolve()
            
            if not file_path.exists():
                print(f"Error: Path does not exist: {file_path}")
                return 1
            
            # Get output directory
            if output_dir:
                dft_output_dir = output_dir / "dft_inputs"
            else:
                dft_output_dir = Path(input("Enter output directory for DFT input files: ") or "./dft_inputs")
            
            dft_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Process the file(s)
            if file_path.is_file():
                # Process a single file with appropriate DFT configuration
                if dft_code == "cp2k":
                    from ..dft.cp2k import interactive_cp2k_config
                    print("\nGenerating CP2K input configuration...")
                    dft_config = interactive_cp2k_config()
                elif dft_code == "vasp":
                    from ..dft.vasp import interactive_vasp_config
                    print("\nGenerating VASP input configuration...")
                    dft_config, _ = interactive_vasp_config()
                elif dft_code == "gaussian":
                    from ..dft.gaussian import interactive_gaussian_config
                    print("\nGenerating Gaussian input configuration...")
                    dft_config = interactive_gaussian_config()
                
                output_file = process_structure_file(file_path, dft_output_dir, dft_code, dft_config)
                if output_file:
                    print(f"\nGenerated DFT input file: {output_file}")
                else:
                    print(f"\nFailed to process file: {file_path}")
            
            elif file_path.is_dir():
                # Process all files in directory
                file_pattern = input("\nEnter file pattern to match (default: *.{xyz,cif}): ") or "*.{xyz,cif}"
                
                # Get DFT configuration
                if dft_code == "cp2k":
                    from ..dft.cp2k import interactive_cp2k_config
                    print("\nGenerating CP2K input configuration...")
                    dft_config = interactive_cp2k_config()
                elif dft_code == "vasp":
                    from ..dft.vasp import interactive_vasp_config
                    print("\nGenerating VASP input configuration...")
                    dft_config, _ = interactive_vasp_config()
                elif dft_code == "gaussian":
                    from ..dft.gaussian import interactive_gaussian_config
                    print("\nGenerating Gaussian input configuration...")
                    dft_config = interactive_gaussian_config()
                
                output_files = process_structure_files(file_path, dft_output_dir, file_pattern, dft_code, dft_config)
                if output_files:
                    print(f"\nGenerated {len(output_files)} DFT input files in {dft_output_dir}")
                else:
                    print(f"\nNo files processed in directory: {file_path}")
        
        return 0
    
    except Exception as e:
        logger.error(f"Error in research command: {e}", exc_info=True)
        print(f"\nError: {e}")
        return 1


def run_process_command(args, config):
    """
    Run the process command.
    
    Args:
        args (argparse.Namespace): Command line arguments.
        config (dict): Configuration dictionary.
    
    Returns:
        int: Exit code.
    """
    input_path = Path(args.input).expanduser().resolve()
    logger.info(f"Running process command for input: {input_path}")
    
    # Load DFT configuration
    dft_config = load_dft_config(args.dft)
    
    try:
        if input_path.is_file():
            # Process a single file
            logger.info(f"Processing file: {input_path}")
            
            output_file = process_structure_file(
                input_path,
                args.output,
                args.dft,
                dft_config
            )
            
            if output_file:
                print(f"\nGenerated DFT input file: {output_file}")
                return 0
            else:
                print(f"\nFailed to process file: {input_path}")
                return 1
        
        elif input_path.is_dir():
            # Process a directory
            logger.info(f"Processing directory: {input_path}")
            
            output_files = process_structure_files(
                input_path,
                args.output,
                args.pattern,
                args.dft,
                dft_config
            )
            
            if output_files:
                print(f"\nGenerated {len(output_files)} DFT input files")
                return 0
            else:
                print(f"\nNo files processed in directory: {input_path}")
                return 1
        
        else:
            logger.error(f"Input path does not exist: {input_path}")
            print(f"\nInput path does not exist: {input_path}")
            return 1
    
    except Exception as e:
        logger.error(f"Error in process command: {e}", exc_info=True)
        print(f"\nError: {e}")
        return 1


def run_interactive_session(config):
    """
    Run an interactive session.
    
    Args:
        config (dict): Configuration dictionary.
    
    Returns:
        int: Exit code.
    """
    from ..dft.cp2k import interactive_cp2k_config
    from ..dft.vasp import interactive_vasp_config
    from ..dft.gaussian import interactive_gaussian_config
    
    print("\nWelcome to the Multi-Agent DFT Research System Interactive Session\n")
    
    try:
        # Initialize agents
        print("Initializing agents...")
        exp_agent = ExperimentalChemistAgent(config=config)
        theo_agent = TheoreticalChemistAgent(config=config)
        sup1_agent = SupervisorAgent("Integration", config=config)
        gaussian_expert = GaussianExpertAgent(config=config)
        vasp_expert = VASPExpertAgent(config=config)
        cp2k_expert = CP2KExpertAgent(config=config)
        sup2_agent = SupervisorAgent("DFT_Recommendation", config=config)
        
        while True:
            print("\n" + "="*50)
            print("Available commands:")
            print("1. Research")
            print("2. Process structure files")
            print("3. Generate DFT input")
            print("4. Publication search")
            print("0. Exit")
            
            choice = input("\nEnter your choice (0-4): ")
            
            if choice == "0":
                print("\nGoodbye!")
                return 0
            
            elif choice == "1":
                # Research mode
                query = input("\nEnter your research topic or question: ")
                max_results = int(input("Maximum number of publications to analyze (default 10): ") or "10")
                
                followup_question = sup1_agent.generate_followup_question(query)
                answer = input(f"\n{followup_question}\nYour answer (press Enter to skip): ")
                additional_context = f"{followup_question} Answer: {answer}\n" if answer else ""
                
                # Refine the query
                refined_query = f"{query} with context: {additional_context}" if additional_context else query
                print(f"\nRefined research query:\n{refined_query}\n")
                
                # Get experimental and theoretical summaries
                print("Getting experimental summary...")
                exp_summary = exp_agent.summarize(query, additional_context, max_results)
                print("Getting theoretical summary...")
                theo_summary = theo_agent.summarize(query, additional_context, max_results)
                
                print("\n--- Experimental Chemist Summary ---")
                print(exp_summary)
                print("\n--- Theoretical Chemist Summary ---")
                print(theo_summary)
                
                # Integrate experimental and theoretical summaries
                print("Integrating summaries...")
                integrated_content = f"Experimental Summary:\n{exp_summary}\n\nTheoretical Summary:\n{theo_summary}"
                sup1_report = sup1_agent.integrate(integrated_content)
                
                print("\n--- Supervisor 1 Integrated Report ---")
                print(sup1_report)
                
                # Get DFT expert reports
                print("\nEngaging DFT Expert Agents...\n")
                
                print("Getting GAUSSIAN expert report...")
                gaussian_report = gaussian_expert.analyze(refined_query)
                print("Getting VASP expert report...")
                vasp_report = vasp_expert.analyze(refined_query)
                print("Getting CP2K expert report...")
                cp2k_report = cp2k_expert.analyze(refined_query)
                
                print("\n--- GAUSSIAN Expert Report ---")
                print(gaussian_report)
                print("\n--- VASP Expert Report ---")
                print(vasp_report)
                print("\n--- CP2K Expert Report ---")
                print(cp2k_report)
                
                # Get final DFT recommendation
                print("Getting DFT recommendation...")
                dft_content = f"GAUSSIAN Report:\n{gaussian_report}\n\nVASP Report:\n{vasp_report}\n\nCP2K Report:\n{cp2k_report}"
                sup2_report = sup2_agent.integrate(dft_content)
                
                print("\n--- Supervisor 2 Final Recommendation ---")
                print(sup2_report)
                
                # Ask if the user wants to save the reports
                save_reports = input("\nDo you want to save the reports? (y/n): ").lower() == 'y'
                if save_reports:
                    output_dir = Path(input("Enter output directory: ") or "./research_output")
                    output_dir.mkdir(parents=True, exist_ok=True)
                    
                    with open(output_dir / "experimental_summary.md", "w") as f:
                        f.write(f"# Experimental Summary for: {query}\n\n{exp_summary}")
                    
                    with open(output_dir / "theoretical_summary.md", "w") as f:
                        f.write(f"# Theoretical Summary for: {query}\n\n{theo_summary}")
                    
                    with open(output_dir / "integrated_report.md", "w") as f:
                        f.write(f"# Integrated Report for: {query}\n\n{sup1_report}")
                    
                    with open(output_dir / "dft_recommendation.md", "w") as f:
                        f.write(f"# DFT Recommendation for: {query}\n\n{sup2_report}")
                    
                    # Save individual DFT reports
                    dft_dir = output_dir / "dft_reports"
                    dft_dir.mkdir(exist_ok=True)
                    
                    with open(dft_dir / "gaussian_report.md", "w") as f:
                        f.write(f"# GAUSSIAN Expert Report for: {query}\n\n{gaussian_report}")
                    
                    with open(dft_dir / "vasp_report.md", "w") as f:
                        f.write(f"# VASP Expert Report for: {query}\n\n{vasp_report}")
                    
                    with open(dft_dir / "cp2k_report.md", "w") as f:
                        f.write(f"# CP2K Expert Report for: {query}\n\n{cp2k_report}")
                    
                    print(f"\nReports saved to: {output_dir}")
            
            elif choice == "2":
                # Process structure files
                input_path = input("\nEnter path to the structure file or directory: ")
                input_path = Path(input_path).expanduser().resolve()
                
                if not input_path.exists():
                    print(f"Error: Path does not exist: {input_path}")
                    continue
                
                output_dir = input("Enter output directory (press Enter for default): ") or None
                dft_code = input("DFT code to use (cp2k/vasp/gaussian) [default: cp2k]: ") or "cp2k"
                
                if input_path.is_file():
                    # Process a single file
                    output_file = process_structure_file(
                        input_path,
                        output_dir,
                        dft_code,
                        load_dft_config(dft_code)
                    )
                    
                    if output_file:
                        print(f"\nGenerated DFT input file: {output_file}")
                    else:
                        print(f"\nFailed to process file: {input_path}")
                
                elif input_path.is_dir():
                    # Process a directory
                    pattern = input("File pattern to match (default: *.{xyz,cif}): ") or "*.{xyz,cif}"
                    
                    output_files = process_structure_files(
                        input_path,
                        output_dir,
                        pattern,
                        dft_code,
                        load_dft_config(dft_code)
                    )
                    
                    if output_files:
                        print(f"\nGenerated {len(output_files)} DFT input files")
                    else:
                        print(f"\nNo files processed in directory: {input_path}")
            
            elif choice == "3":
                # Generate DFT input interactively
                dft_code = input("\nWhich DFT code would you like to use? (cp2k/vasp/gaussian) [default: cp2k]: ") or "cp2k"
                
                if dft_code.lower() == "cp2k":
                    print("\nGenerating CP2K input configuration...")
                    cp2k_config = interactive_cp2k_config()
                    
                    file_type = input("Do you have CIF files or XYZ files? (Enter CIF/XYZ): ")
                    file_dir = input("Enter the directory path containing your structure files: ")
                    out_dir = input("Enter the output directory path where input files will be generated: ")
                    
                    if file_type.lower() == "cif":
                        process_structure_files(file_dir, out_dir, "*.cif", "cp2k", cp2k_config)
                    elif file_type.lower() == "xyz":
                        process_structure_files(file_dir, out_dir, "*.xyz", "cp2k", cp2k_config)
                    else:
                        print("Invalid file type.")
                
                elif dft_code.lower() == "vasp":
                    print("\nGenerating VASP input configuration...")
                    incar_config, kpoints_config = interactive_vasp_config()
                    
                    file_dir = input("Enter the directory path containing your structure files: ")
                    out_dir = input("Enter the output directory path where input files will be generated: ")
                    
                    process_structure_files(file_dir, out_dir, "*.{xyz,cif}", "vasp", incar_config)
                
                elif dft_code.lower() == "gaussian":
                    print("\nGenerating Gaussian input configuration...")
                    gaussian_config = interactive_gaussian_config()
                    
                    file_dir = input("Enter the directory path containing your structure files: ")
                    out_dir = input("Enter the output directory path where input files will be generated: ")
                    
                    process_structure_files(file_dir, out_dir, "*.{xyz,cif}", "gaussian", gaussian_config)
                
                else:
                    print(f"Unsupported DFT code: {dft_code}")
            
            elif choice == "4":
                # Publication search
                pub_api = PublicationAPI(config=config)
                
                query = input("\nEnter your search query: ")
                max_results = int(input("Maximum number of results (default 10): ") or "10")
                
                print(f"\nSearching for publications related to: {query}")
                publications = pub_api.search(query, max_results=max_results)
                
                if not publications:
                    print("No publications found.")
                    continue
                
                print(f"\nFound {len(publications)} publications.")
                
                # Ask for keywords to analyze
                keywords = input("\nEnter keywords to analyze (comma-separated, press Enter to skip): ")
                if keywords:
                    keywords = [k.strip() for k in keywords.split(",")]
                    analysis = pub_api.analyze_publications(publications, keywords)
                else:
                    analysis = pub_api.analyze_publications(publications)
                
                # Generate report
                report = pub_api.generate_report(publications, analysis)
                print("\n" + "="*50)
                print(report)
                
                # Ask if the user wants to save the report
                save_report = input("\nDo you want to save the report? (y/n): ").lower() == 'y'
                if save_report:
                    output_dir = Path(input("Enter output directory: ") or "./publication_reports")
                    output_dir.mkdir(parents=True, exist_ok=True)
                    
                    report_file = output_dir / f"publication_report_{query.replace(' ', '_')}.md"
                    with open(report_file, "w") as f:
                        f.write(report)
                    
                    print(f"\nReport saved to: {report_file}")
            
            else:
                print("\nInvalid choice. Please try again.")
    
    except KeyboardInterrupt:
        print("\n\nSession terminated by user.")
        return 0
    
    except Exception as e:
        logger.error(f"Error in interactive session: {e}", exc_info=True)
        print(f"\nError: {e}")
        return 1


def show_version():
    """
    Show version information.
    
    Returns:
        int: Exit code.
    """
    from .. import __version__
    print(f"Multi-Agent DFT Research System v{__version__}")
    print("(c) 2023")
    return 0


def main():
    """
    Main entry point for the CLI.
    
    Returns:
        int: Exit code.
    """
    args = parse_args()
    
    # Setup logging
    log_file = args.log_file or get_default_log_file()
    setup_logging(args.log_level, log_file, not args.no_console)
    
    # Log system information
    log_system_info()
    
    # Load configuration
    try:
        if args.config:
            config = load_config(args.config)
            logger.info(f"Loaded configuration from {args.config}")
        else:
            config = load_config()
            logger.info("Loaded default configuration")
    
    except Exception as e:
        logger.error(f"Error loading configuration: {e}", exc_info=True)
        print(f"Error loading configuration: {e}")
        return 1
    
    # Run the appropriate command
    if args.command == "research":
        return run_research_command(args, config)
    
    elif args.command == "process":
        return run_process_command(args, config)
    
    elif args.command == "interactive":
        return run_interactive_session(config)
    
    elif args.command == "version":
        return show_version()
    
    else:
        # If no command or unknown command, run interactive session
        return run_interactive_session(config)


if __name__ == "__main__":
    sys.exit(main())