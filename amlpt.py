"""
Multi-Agent DFT Research System - Main Entry Point

This script initializes and runs the Multi-Agent DFT Research System, which orchestrates
a multi-agent workflow to address research queries, perform literature analysis, and
process crystallographic structure files for input generation for DFT simulations.
"""
# -*- coding: utf-8 -*-
import os
import sys
import json
import glob
import math 
import yaml  # You'll need this for YAML file handling if you keep that option
from pathlib import Path
from typing import List, Dict, Any, Optional  # ADD THIS LINE
import numpy as np
from collections import Counter 
from multi_agent_dft.utils.logging import get_logger
from multi_agent_dft.utils.validator import validate_structure
from multi_agent_dft.utils.converter import convert_input_format
from multi_agent_dft.agents.chemistry_agents import ExperimentalChemistAgent, TheoreticalChemistAgent
from multi_agent_dft.agents.dft_agents       import GaussianExpertAgent, VASPExpertAgent, CP2KExpertAgent
from multi_agent_dft.agents.supervisor       import SupervisorAgent
from multi_agent_dft.file_processing.dft_output_processor import process_dft_output
from multi_agent_dft.file_processing.dft_output_processor import VASPOutputProcessor
from multi_agent_dft.file_processing.ml_dataset_converter import create_mace_h5_dataset

# FIXED: Complete the VASP MD processor imports
from multi_agent_dft.file_processing.vasp_md_processor import (
    check_ase_available,
    extract_final_structure_from_json,
    write_complete_vasp_md_input,
    structure_file_to_atoms,
    get_vasp_md_template
)

from multi_agent_dft.file_processing.cif import process_cif_files, cif_to_xyz

try:
    from ase import Atoms
    from ase.geometry import cellpar_to_cell, cell_to_cellpar
    ASE_AVAILABLE = True
except ImportError:
    ASE_AVAILABLE = False


logger = get_logger(__name__)

class MultiAgentSystem:
    def __init__(self, config_path=None):
        logger.info("Initializing Multi-Agent DFT Research System")
        self.config        = __import__('multi_agent_dft.config', fromlist=['load_config']).load_config(config_path) \
                               if config_path else {}
        self.exp_agent     = ExperimentalChemistAgent   (config=self.config)
        self.theo_agent    = TheoreticalChemistAgent     (config=self.config)
        self.sup1_agent    = SupervisorAgent("Integration",        config=self.config)
        self.gaussian_expert = GaussianExpertAgent      (config=self.config)
        self.vasp_expert     = VASPExpertAgent          (config=self.config)
        self.cp2k_expert     = CP2KExpertAgent          (config=self.config)
        self.sup2_agent    = SupervisorAgent("DFT_Recommendation", config=self.config)

    def run(self):
        print("\n" + "="*80)
        print("  Welcome to the Machine Learning Potential Research System  ".center(80, "="))
        print("="*80 + "\n")

        # Ask for mode selection
        print("Please select an operation mode:")
        print("1. AI-agent feedback (summaries & reports)")
        print("2. Input generation (CP2K/VASP/Gaussian)")
        print("3. Output processing (extract forces, energies, coordinates)")
        print("4. ML potential dataset creation (JSON to MACE HDF5)")
        print("5. AIMD processing (JSON to CP2K/VASP AIMD inputs)")  # MODIFIED
        print("6. VASP MD input generation")  # NEW OPTION
        
        mode = input("\nEnter your choice (1/2/3/4/5/6): ").strip()
        
        if mode == "1":
            # Run AI-assisted research workflow
            self._run_ai_workflow()
            # After AI workflow, ask if user wants to proceed to input generation
            proceed = input("\nWould you like to proceed to input generation? (y/n): ").strip().lower()
            if proceed == "y":
                self._choose_input_mode()
        elif mode == "3":
            # Run output processing workflow
            self._handle_output_processing()
        elif mode == "4":
            # Run ML potential dataset creation workflow
            self._handle_ml_dataset_creation()
        elif mode == "5":
            # Run AIMD processing workflow (now supports both CP2K and VASP)
            self._handle_aimd_processing()
        elif mode == "6":
            # NEW: VASP MD input generation
            self._handle_vasp_md_input_generation()
        else:
            # Default to input generation
            self._choose_input_mode()


########## AI AGENT METHODS ##########
    def _refine_query(self, q, ctx):
        msgs = [
            {"role":"system","content":
                "You are a research query specialist. Refine the query to be specific, comprehensive and aligned with current terminology."
            },
            {"role":"user","content":
                f"Original query: {q}\nAdditional context: {ctx}\n\nPlease refine for optimal search."
            }
        ]
        return self.sup1_agent.chat(msgs)
    def _run_ai_workflow(self):
        """Run the AI-assisted research workflow with enhanced reporting."""
        user_query = input("Enter your research topic or question: ").strip()
        followup   = self.sup1_agent.generate_followup_question(user_query)
        ans        = input(f"\n{followup}\nYour answer (press Enter to skip): ").strip()
        ctx        = f"{followup} Answer: {ans}" if ans else ""

        refined = self._refine_query(user_query, ctx)
        print(f"\nRefined query:\n{refined}\n")

        print("Analyzing literature...\n")
        
        # Get summaries with references from chemistry agents
        exp_result = self.exp_agent.summarize_with_references(refined, ctx)
        theo_result = self.theo_agent.summarize_with_references(refined, ctx)

        print("\n--- Experimental Chemist Summary ---")
        print(exp_result['summary'])
        print("\n--- Theoretical Chemist Summary ---")
        print(theo_result['summary'])

        integrated = f"Experimental:\n{exp_result['summary']}\n\nTheoretical:\n{theo_result['summary']}"
        sup1_rep   = self.sup1_agent.integrate(integrated)
        print("\n--- Integrated Report ---")
        print(sup1_rep)

        print("\nEngaging DFT experts...\n")
        
        # Get expert analyses with references
        print("--- Gaussian Expert Analysis ---")
        try:
            gauss_result = self.gaussian_expert.analyze_with_references(refined)
        except AttributeError:
            # Fallback to old method if new method not implemented
            gauss_analysis = self.gaussian_expert.analyze(refined)
            gauss_result = {'summary': gauss_analysis, 'references': []}
        print(gauss_result['summary'])
        print()

        print("--- VASP Expert Analysis ---")
        try:
            vasp_result = self.vasp_expert.analyze_with_references(refined)
        except AttributeError:
            # Fallback to old method if new method not implemented
            vasp_analysis = self.vasp_expert.analyze(refined)
            vasp_result = {'summary': vasp_analysis, 'references': []}
        print(vasp_result['summary'])
        print()

        print("--- CP2K Expert Analysis ---")
        try:
            cp2k_result = self.cp2k_expert.analyze_with_references(refined)
        except AttributeError:
            # Fallback to old method if new method not implemented
            cp2k_analysis = self.cp2k_expert.analyze(refined)
            cp2k_result = {'summary': cp2k_analysis, 'references': []}
        print(cp2k_result['summary'])
        print()

        dft_content = f"Gaussian:\n{gauss_result['summary']}\n\nVASP:\n{vasp_result['summary']}\n\nCP2K:\n{cp2k_result['summary']}"
        dft_rec = self.sup2_agent.integrate(dft_content)
        print("\n--- DFT Recommendation ---")
        print(dft_rec)

        # Create reports directory with timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        reports_dir = Path.cwd() / f"reports_{timestamp}"
        reports_dir.mkdir(exist_ok=True)
        
        print(f"\nCreating reports in: {reports_dir}")

        # Format and save reports with references
        reports = {
            "experimental_chemist_report.txt": {
                "content": exp_result['summary'],
                "references": exp_result.get('references', []),
                "title": "Experimental Chemistry Analysis"
            },
            "theoretical_chemist_report.txt": {
                "content": theo_result['summary'],
                "references": theo_result.get('references', []),
                "title": "Theoretical Chemistry Analysis"
            },
            "supervisor_report.txt": {
                "content": sup1_rep,
                "references": [],
                "title": "Integrated Scientific Report"
            },
            "gaussian_expert_report.txt": {
                "content": gauss_result['summary'],
                "references": gauss_result.get('references', []),
                "title": "Gaussian DFT Expert Analysis"
            },
            "vasp_expert_report.txt": {
                "content": vasp_result['summary'],
                "references": vasp_result.get('references', []),
                "title": "VASP DFT Expert Analysis"
            },
            "cp2k_expert_report.txt": {
                "content": cp2k_result['summary'],
                "references": cp2k_result.get('references', []),
                "title": "CP2K DFT Expert Analysis"
            },
            "dft_recommendation_report.txt": {
                "content": dft_rec,
                "references": [],
                "title": "DFT Method Recommendation"
            }
        }

        # Write formatted reports
        for filename, report_data in reports.items():
            formatted_report = self._format_report_with_references(
                title=report_data['title'],
                content=report_data['content'],
                references=report_data['references'],
                query=user_query,
                timestamp=timestamp
            )
            
            report_path = reports_dir / filename
            report_path.write_text(formatted_report, encoding="utf-8")
            print(f"  ✓ {filename}")

        # Create a summary index file
        index_content = self._create_report_index(user_query, timestamp, list(reports.keys()))
        (reports_dir / "README.md").write_text(index_content, encoding="utf-8")
        print(f"  ✓ README.md (index)")

        print(f"\nAll reports saved in: {reports_dir}")
        print(f"Open README.md for an overview of all reports.")

        # After AI workflow, ask if user wants to proceed to input generation
        proceed = input("\nWould you like to proceed to input generation? (y/n): ").strip().lower()
        if proceed == "y":
            self._choose_input_mode()

    def _format_report_with_references(self, title, content, references, query, timestamp):
        """Format a report with proper scientific references."""
        from datetime import datetime
        
        # Header
        report_lines = [
            "=" * 80,
            f"{title.upper()}",
            "=" * 80,
            "",
            f"Research Query: {query}",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Report ID: {timestamp}",
            "",
            "ABSTRACT",
            "-" * 40,
            ""
        ]
        
        # Add content
        report_lines.append(content)
        report_lines.append("")
        
        # Add references section if there are any
        if references:
            report_lines.extend([
                "",
                "REFERENCES",
                "-" * 40,
                ""
            ])
            
            for i, ref in enumerate(references, 1):
                # Format reference in scientific style
                authors = ref.get('authors', ['Unknown'])
                if isinstance(authors, list):
                    if len(authors) > 3:
                        author_str = f"{authors[0]} et al."
                    else:
                        author_str = ", ".join(authors)
                else:
                    author_str = str(authors)
                
                title = ref.get('title', 'Untitled')
                journal = ref.get('journal', 'Unknown Journal')
                year = ref.get('year', 'Unknown')
                url = ref.get('url', '')
                
                reference_line = f"[{i}] {author_str} ({year}). {title}. {journal}."
                if url:
                    reference_line += f" Available at: {url}"
                
                report_lines.append(reference_line)
                report_lines.append("")
        else:
            report_lines.extend([
                "",
                "REFERENCES",
                "-" * 40,
                "",
                "No specific literature references were used in generating this report.",
                "This analysis was based on the integrated findings from the research workflow.",
                ""
            ])
        
        # Footer
        report_lines.extend([
            "",
            "=" * 80,
            f"End of Report - Generated by Multi-Agent DFT Research System",
            "=" * 80
        ])
        
        return "\n".join(report_lines)

    def _create_report_index(self, query, timestamp, report_files):
        """Create an index/README file for the reports directory."""
        from datetime import datetime
        
        index_lines = [
            "# Multi-Agent DFT Research System - Report Summary",
            "",
            f"**Research Query:** {query}",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Report ID:** {timestamp}",
            "",
            "## Overview",
            "",
            "This directory contains a comprehensive analysis of your research query using our multi-agent system.",
            "Each report provides specialized insights from different perspectives in computational chemistry.",
            "",
            "## Report Files",
            "",
            "| File | Description |",
            "|------|-------------|",
        ]
        
        # Map filenames to descriptions
        descriptions = {
            "experimental_chemist_report.txt": "Analysis from experimental chemistry perspective, focusing on synthesis and characterization methods",
            "theoretical_chemist_report.txt": "Analysis from theoretical chemistry perspective, focusing on computational methods and models", 
            "supervisor_report.txt": "Integrated analysis combining experimental and theoretical insights",
            "gaussian_expert_report.txt": "Specialized analysis for Gaussian quantum chemistry software recommendations",
            "vasp_expert_report.txt": "Specialized analysis for VASP density functional theory software recommendations",
            "cp2k_expert_report.txt": "Specialized analysis for CP2K quantum chemistry software recommendations",
            "dft_recommendation_report.txt": "Final recommendations for DFT computational approaches"
        }
        
        for filename in report_files:
            description = descriptions.get(filename, "Specialized analysis report")
            index_lines.append(f"| `{filename}` | {description} |")
        
        index_lines.extend([
            "",
            "## How to Use These Reports",
            "",
            "1. **Start with `supervisor_report.txt`** - This provides an integrated overview",
            "2. **Review specialist reports** - Dive into experimental or theoretical details as needed",
            "3. **Check DFT recommendations** - Use the expert reports for computational method selection",
            "4. **Follow references** - Each report includes numbered references for further reading",
            "",
            "## Report Structure",
            "",
            "Each report follows a standard academic format:",
            "- **Header**: Query, timestamp, and report identification",
            "- **Abstract/Content**: Main analysis and findings", 
            "- **References**: Numbered citations in scientific format",
            "",
            "## Next Steps",
            "",
            "Based on these reports, you may want to:",
            "- Generate DFT input files using the system's input generation features",
            "- Explore specific computational parameters mentioned in the expert reports",
            "- Follow up on literature references for deeper understanding",
            "",
            "---",
            f"*Generated by Multi-Agent DFT Research System on {datetime.now().strftime('%Y-%m-%d')}*"
        ])
        
        return "\n".join(index_lines)



####### INPUT GENERATION METHODS ##########
    def _choose_input_mode(self, use_supercell=None, supercell_dims=None):
        """Ask user: batch‐mode or guided interactive mode."""
        # Only configure supercell if parameters weren't already passed in
        if use_supercell is None and supercell_dims is None:
            use_supercell, supercell_dims = self._configure_supercell()
        
        print("Input‐generation modes:")
        print("  1) Batch‐mode (auto‐convert every file using default template)")
        print("  2) Guided‐mode (step through CP2K/VASP/Gaussian handlers)")
        choice = input("Select mode (1/2, default=2): ").strip() or "2"
        if choice == "1":
            self._batch_input_generation(use_supercell=use_supercell, supercell_dims=supercell_dims)
        else:
            self._handle_simulation_input_generation(use_supercell=use_supercell, supercell_dims=supercell_dims)

    def _configure_supercell(self):
        """Ask user if they want to create a supercell and get dimensions."""
        print("\n==== Supercell Configuration ====")
        use_supercell = input("Create a supercell? (y/n) [n]: ").strip().lower() or "n"
        
        if use_supercell != "y":
            return False, None
        
        print("\nEnter supercell dimensions as multipliers for each axis:")
        try:
            nx = int(input("Multiplier for x-axis [1]: ").strip() or "1")
            ny = int(input("Multiplier for y-axis [1]: ").strip() or "1")
            nz = int(input("Multiplier for z-axis [1]: ").strip() or "1")
            
            if nx <= 0 or ny <= 0 or nz <= 0:
                print("Invalid values. All dimensions must be positive integers.")
                print("Using default 1x1x1 supercell.")
                return True, (1, 1, 1)
            
            print(f"\nSupercell dimensions: {nx}x{ny}x{nz}")
            return True, (nx, ny, nz)
        except ValueError:
            print("Invalid input. Using default 1x1x1 supercell.")
            return True, (1, 1, 1)

    def _batch_input_generation(self, use_supercell=False, supercell_dims=None):
        """Auto‐convert all supported files in folder using default templates with improved performance."""
        code = input("Batch‐mode: which DFT code? (CP2K/VASP/Gaussian): ").strip().lower()
        path_input = input("Path to file or directory: ").strip()
        path = Path(path_input).expanduser().resolve()
        out_input = input("Output directory: ").strip() 
        out = Path(out_input).expanduser().resolve()
        out.mkdir(parents=True, exist_ok=True)

        # Check if path exists
        if not path.exists():
            print(f"Error: Path '{path}' does not exist.")
            return
        
        # OPTIMIZATION: Ask for file limit to improve control
        limit_files = input("Limit number of files to process? (y/n) [n]: ").strip().lower() or "n"
        file_limit = None
        if limit_files == "y":
            try:
                file_limit = int(input("Maximum number of files to process: "))
                print(f"Will process up to {file_limit} files")
            except ValueError:
                print("Invalid input. Will process all files.")
        
        # Pick default template
        if code=="cp2k":
            params = self._get_cp2k_template()
            fmt    = "cp2k";  ext=".inp"
        elif code == "vasp":
            params = self._get_vasp_template()
            kpoints = params.pop('KPOINTS', {'type': 'gamma', 'grid': [3, 3, 3]})  # Extract KPOINTS
            fmt = "vasp"
            ext = ""
        else:
            params = self._get_gaussian_params()
            fmt    = "gaussian"; ext=".com"

        # Get the list of files to process
        if path.is_dir():
            # Use recursive glob if the path contains **
            if "**" in path_input:
                cif_files = list(path.glob("**/*.cif"))
                xyz_files = list(path.glob("**/*.xyz"))
            else:
                cif_files = list(path.glob("*.cif"))
                xyz_files = list(path.glob("*.xyz"))
            
            files = cif_files + xyz_files
            
            if not files:
                print(f"No CIF or XYZ files found in {path}")
                return
                
            # OPTIMIZATION: Apply file limit if specified
            if file_limit and len(files) > file_limit:
                files = files[:file_limit]
                print(f"Limited to processing {file_limit} files out of {len(cif_files) + len(xyz_files)} total files")
            
            print(f"Found {len(files)} files to process:")
            for i, f in enumerate(files, 1):
                print(f"  {i}. {f.name}")
        else:
            # For a single file
            if path.suffix.lower() not in ['.cif', '.xyz']:
                print(f"Error: File must be a CIF or XYZ file, got: {path.suffix}")
                return
            
            files = [path]
            print(f"Processing single file: {path.name}")

        # OPTIMIZATION: Ask about k-point scaling for VASP with large supercells
        auto_kpt_scaling = False
        if fmt == "vasp" and use_supercell:
            auto_kpt = input("Automatically scale k-points for large supercells? (y/n) [y]: ").strip().lower() or "y"
            auto_kpt_scaling = (auto_kpt == "y")
            if auto_kpt_scaling:
                print("K-points will be automatically reduced for large supercells")
                
        # OPTIMIZATION: Create a progress indicator
        total_files = len(files)
        print(f"\nStarting batch processing of {total_files} files...")
        
        # OPTIMIZATION: Create temp directory once
        if use_supercell:
            temp_dir = out / "temp_supercell"
            temp_dir.mkdir(exist_ok=True)
        
        processed_count = 0
        for i, f in enumerate(files, 1):
            stem = f.stem.replace(" ","_")
            print(f"\n[{i}/{total_files}] Processing {f.name}...", end=" ")
            
            # Handle supercell creation if requested
            try:
                if use_supercell and supercell_dims:
                    # Create supercell using ASE
                    nx, ny, nz = supercell_dims
                    try:
                        from ase.io import read, write
                        from ase.build import make_supercell
                        import numpy as np
                        
                        # OPTIMIZATION: Inform user but don't repeat supercell details
                        print(f"\nCreating {nx}x{ny}x{nz} supercell...")
                        ase_struct = read(f)
                        NxNxN = np.array([nx, ny, nz])
                        supercell = make_supercell(ase_struct, np.eye(3) * NxNxN)
                        
                        # Save the supercell to a temporary file
                        temp_file = temp_dir / f"{stem}_supercell.xyz"  # Use XYZ for faster processing
                        write(temp_file, supercell)
                        
                        # Update the file to process
                        f = temp_file
                        orig_stem = stem
                        stem = f"{stem}_{nx}x{ny}x{nz}"
                        
                        # OPTIMIZATION: Only show atom count for large structures
                        if len(supercell) > 100:
                            print(f"Supercell created with {len(supercell)} atoms")
                            
                        # OPTIMIZATION: Scale k-points for large supercells in VASP
                        if fmt == "vasp" and auto_kpt_scaling and len(supercell) > 200 and 'grid' in kpoints:
                            original_grid = kpoints['grid'].copy()
                            # Scale k-points inversely with supercell size
                            kpoints['grid'] = [max(1, int(g / max(nx, ny, nz))) for g in original_grid]
                            print(f"Adjusted k-point grid: {original_grid} → {kpoints['grid']}")
                            
                    except Exception as e:
                        print(f"\nError creating supercell: {e}")
                        print(f"Proceeding with original structure")
                        # Revert to original file and stem
                        f = files[i-1]
                        stem = f.stem.replace(" ","_")
                
                # VASP-specific processing
                if fmt == "vasp":
                    # Create a directory for this structure
                    struct_dir = out / stem
                    struct_dir.mkdir(parents=True, exist_ok=True)
                    
                    try:
                        # For CIF files, convert to XYZ first
                        if f.suffix.lower() == '.cif':
                            xyz_path = out / f"{stem}.xyz"
                            if cif_to_xyz(f, xyz_path):
                                self._copy_to_xyz_checks(xyz_path, out)
                                # Include kpoints configuration
                                result = self._generate_vasp_input_for_file(xyz_path, struct_dir, params, kpoints)
                                # OPTIMIZATION: Delete the temporary XYZ file after use
                                if xyz_path.exists():
                                    try:
                                        xyz_path.unlink()
                                    except:
                                        pass
                                print("OK" if result else "ERROR")
                                processed_count += 1 if result else 0
                            else:
                                print("ERROR: CIF conversion failed")
                        else:
                            # Direct XYZ processing
                            result = self._generate_vasp_input_for_file(f, struct_dir, params, kpoints)
                            print("OK" if result else "ERROR")
                            processed_count += 1 if result else 0
                    except Exception as e:
                        print(f"ERROR: {str(e)}")
                else:
                    # Non-VASP processing (CP2K, Gaussian)
                    outp = out/(stem+ext)
                    try:
                        res = convert_input_format(f, fmt, outp, parameters=params)
                        print("OK" if res else "ERROR")
                        processed_count += 1 if res else 0
                    except Exception as e:
                        print(f"ERROR: {str(e)}")
                
                # OPTIMIZATION: Clean up temp files as we go
                if use_supercell and f.parent == temp_dir:
                    try:
                        f.unlink()
                    except:
                        pass
            
            except Exception as e:
                print(f"Unexpected error: {str(e)}")
                
            # OPTIMIZATION: Option to pause or stop batch processing
            if i % 10 == 0 and i < total_files:
                command = input(f"\nProcessed {i}/{total_files} files. Continue, Pause, or Stop? (c/p/s) [c]: ").strip().lower() or "c"
                if command == "s":
                    print("Stopping batch processing at user request.")
                    break
                elif command == "p":
                    input("Paused. Press Enter to continue...")
                    
        # Clean up temporary directory if it was created
        if use_supercell and (out / "temp_supercell").exists():
            import shutil
            try:
                shutil.rmtree(out / "temp_supercell")
            except:
                pass
            
        print(f"\nBatch processing completed. Successfully processed {processed_count} out of {len(files)} files.")
        
        # Verify results
        if fmt == "vasp":
            self._verify_vasp_output_files(out)

    def _handle_simulation_input_generation(self, use_supercell=False, supercell_dims=None):
        """Guided CP2K/VASP/Gaussian handlers."""
        code = input("Which DFT code? (CP2K/VASP/Gaussian): ").strip().lower()
        if code=="cp2k":
            self._handle_cp2k_input_generation(use_supercell=use_supercell, supercell_dims=supercell_dims)
        elif code=="vasp":
            self._handle_vasp_input_generation(use_supercell=use_supercell, supercell_dims=supercell_dims)
        elif code=="gaussian":
            self._handle_gaussian_input_generation(use_supercell=use_supercell, supercell_dims=supercell_dims)
        else:
            print("Unsupported code.")

####### CP2K INPUT GENERATION METHODS ##########

    def _handle_cp2k_input_generation(self, use_supercell=False, supercell_dims=None):
        """Handle CP2K input generation with detailed step-by-step guidance."""
        print("\n==== CP2K Input File Generation (Guided Mode) ====\n")
        
        # Step 1: Get structure file path
        file_path = input("Enter the path to your structure file or directory (XYZ or CIF): ")
        input_path = Path(file_path).expanduser().resolve()
        
        # Validate the input file/directory
        if not input_path.exists():
            print(f"Error: Path '{input_path}' does not exist.")
            return
        
        # Step 2: Output location
        output_path = input("Enter the output directory path where input files will be generated: ")
        output_dir = Path(output_path).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Step 3: If directory - ask for batch processing or select specific file
        if input_path.is_dir():
            cif_files = list(input_path.glob("*.cif"))
            xyz_files = list(input_path.glob("*.xyz"))
            
            if not (cif_files or xyz_files):
                print(f"No .cif or .xyz files found in {input_path}")
                return
            
            print(f"\nFound {len(cif_files)} CIF files and {len(xyz_files)} XYZ files.")
            process_mode = input("Process all files or select a specific file? (all/select): ").strip().lower()
            
            if process_mode == "select":
                # Show files for selection
                all_files = cif_files + xyz_files
                for i, f in enumerate(all_files, 1):
                    print(f"{i}. {f.name}")
                
                try:
                    selected = int(input("\nSelect file number: "))
                    if 1 <= selected <= len(all_files):
                        input_path = all_files[selected-1]
                    else:
                        print("Invalid selection. Using first file.")
                        input_path = all_files[0]
                except ValueError:
                    print("Invalid input. Using first file.")
                    input_path = all_files[0]
                    
                print(f"\nSelected file: {input_path.name}")
                
                # Now input_path is a single file
                # Process with supercell if requested
                if use_supercell and supercell_dims:
                    self._process_supercell_cp2k_file(input_path, output_dir, supercell_dims)
                else:
                    self._process_single_cp2k_file(input_path, output_dir)
            else:
                # Proceed with batch processing
                print("\nBatch processing all structure files...\n")
                
                # Step 4: Get CP2K parameters (once for all files)
                cp2k_config = self._get_cp2k_parameters_interactively()
                
                # Process all files
                processed = 0
                
                if cif_files:
                    print(f"\nProcessing {len(cif_files)} CIF files...")
                    for cif_file in cif_files:
                        try:
                            print(f"  Processing {cif_file.name}...")
                            
                            # Handle supercell creation if requested
                            if use_supercell and supercell_dims:
                                self._process_supercell_cp2k_file(cif_file, output_dir, supercell_dims, cp2k_config)
                            else:
                                # Original processing
                                xyz_path = output_dir / f"{cif_file.stem}.xyz"
                                if cif_to_xyz(cif_file, xyz_path):
                                    self._copy_to_xyz_checks(xyz_path, output_dir)
                                    processed += 1
                                    self._generate_cp2k_input_for_file(xyz_path, output_dir, cp2k_config)
                            
                            processed += 1
                        except Exception as e:
                            print(f"  Error processing {cif_file.name}: {e}")
                
                if xyz_files:
                    print(f"\nProcessing {len(xyz_files)} XYZ files...")
                    for xyz_file in xyz_files:
                        try:
                            print(f"  Processing {xyz_file.name}...")
                            
                            # Handle supercell creation if requested
                            if use_supercell and supercell_dims:
                                self._process_supercell_cp2k_file(xyz_file, output_dir, supercell_dims, cp2k_config)
                            else:
                                # Original processing
                                self._copy_to_xyz_checks(xyz_file, output_dir)
                                processed += 1
                                self._generate_cp2k_input_for_file(xyz_file, output_dir, cp2k_config)
                        except Exception as e:
                            print(f"  Error processing {xyz_file.name}: {e}")
                
                print(f"\nProcessed {processed} files. CP2K input files generated in {output_dir}")
        else:
            # Process single file
            if use_supercell and supercell_dims:
                self._process_supercell_cp2k_file(input_path, output_dir, supercell_dims)
            else:
                self._process_single_cp2k_file(input_path, output_dir)

        self._cleanup_xyz_files(output_dir)
        self._verify_output_files(output_dir)

    def _process_single_cp2k_file(self, file_path, output_dir):
        """Process a single file for CP2K input generation."""
        # Validate the file
        valid, msg, _ = validate_structure(file_path)
        if not valid:
            print(f"Structure validation failed: {msg}")
            return
        
        # For CIF files, convert to XYZ first
        if file_path.suffix.lower() == ".cif":
            print(f"\nConverting CIF file to XYZ format...")
            xyz_path = output_dir / f"{file_path.stem}.xyz"
            if not cif_to_xyz(file_path, xyz_path):
                print("CIF conversion failed. Aborting.")
                return
            file_path = xyz_path
            print(f"Conversion successful: {xyz_path}")
            self._copy_to_xyz_checks(xyz_path, output_dir)
        
        # Get CP2K parameters through interactive prompts
        cp2k_config = self._get_cp2k_parameters_interactively()
        
        # Generate CP2K input file
        self._generate_cp2k_input_for_file(file_path, output_dir, cp2k_config)
        
        print(f"\nCP2K input generated in {output_dir}")

    def _process_supercell_cp2k_file(self, file_path, output_dir, supercell_dims, cp2k_config=None):
        """Process a single file with supercell creation for CP2K input generation."""
        nx, ny, nz = supercell_dims
        
        # Validate the file
        valid, msg, _ = validate_structure(file_path)
        if not valid:
            print(f"Structure validation failed: {msg}")
            return
        
        # Create a temporary directory for supercell processing
        temp_dir = output_dir / "temp_supercell"
        temp_dir.mkdir(exist_ok=True)
        
        try:
            from ase.io import read, write
            from ase.build import make_supercell
            import numpy as np
            
            # For CIF files, convert to XYZ first if needed
            if file_path.suffix.lower() == ".cif":
                print(f"\nConverting CIF file to XYZ format...")
                xyz_path = temp_dir / f"{file_path.stem}.xyz"
                if not cif_to_xyz(file_path, xyz_path):
                    print("CIF conversion failed. Aborting.")
                    return
                file_path = xyz_path
                print(f"Conversion successful: {xyz_path}")
                self._copy_to_xyz_checks(xyz_path, output_dir)
            
            # Create the supercell
            print(f"Creating {nx}x{ny}x{nz} supercell...")
            ase_struct = read(file_path)
            NxNxN = np.array([nx, ny, nz])
            supercell = make_supercell(ase_struct, np.eye(3) * NxNxN)
            
            # Save the supercell to a temporary file
            supercell_file = temp_dir / f"{file_path.stem}_{nx}x{ny}x{nz}.xyz"
            write(supercell_file, supercell)
            print(f"Supercell created with {len(supercell)} atoms")
            
            # Get CP2K parameters if not provided
            if cp2k_config is None:
                cp2k_config = self._get_cp2k_parameters_interactively()
            
            # Generate CP2K input for the supercell
            output_file = output_dir / f"{file_path.stem}_{nx}x{ny}x{nz}.inp"
            
            # Parse XYZ file and generate CP2K input
            from multi_agent_dft.file_processing.xyz import parse_xyz
            from multi_agent_dft.dft.cp2k import save_cp2k_input
            
            structure_data = parse_xyz(supercell_file)
            if structure_data:
                success = save_cp2k_input(structure_data, output_file, cp2k_config)
                if success:
                    print(f"Generated CP2K input file: {output_file}")
                else:
                    print(f"Failed to generate CP2K input file for {supercell_file}")
            else:
                print(f"Error: Could not parse XYZ file {supercell_file}")
            
        except Exception as e:
            print(f"Error in supercell creation or CP2K input generation: {e}")
        finally:
            # Clean up temporary directory
            import shutil
            shutil.rmtree(temp_dir)

    def _generate_cp2k_input_for_file(self, xyz_file, output_dir, cp2k_config):
        """Generate a CP2K input file for a single XYZ file."""
        from multi_agent_dft.file_processing.xyz import parse_xyz
        from multi_agent_dft.dft.cp2k import save_cp2k_input
        
        try:
            # Parse XYZ file
            structure_data = parse_xyz(xyz_file)
            if not structure_data:
                print(f"Error: Could not parse XYZ file {xyz_file}")
                return False
            
            # Generate CP2K input file
            output_file = output_dir / f"{xyz_file.stem}.inp"
            success = save_cp2k_input(structure_data, output_file, cp2k_config)
            
            if success:
                print(f"  Generated CP2K input file: {output_file}")
                return True
            else:
                print(f"  Failed to generate CP2K input file for {xyz_file}")
                return False
        except Exception as e:
            print(f"  Error generating CP2K input for {xyz_file}: {e}")
            return False

    def _get_cp2k_parameters_interactively(self):
        """Interactive questionnaire for CP2K input parameters."""
        print("\n=== CP2K Calculation Settings ===")
        
        # Initialize config
        cp2k_config = {
            'global': {},
            'dft': {'xc': {}},
            'scf': {},
            'mgrid': {},
            'qs': {},
            'motion': {},
            'print': {},
            'subsys': {'cell': {}}
        }
        
        # 1. Project name and run type
        print("\n--- Basic Settings ---")
        project = input("Project name [cp2k_project]: ").strip() or "cp2k_project"
        cp2k_config['global']['PROJECT'] = project
        
        print("\nAvailable run types:")
        run_types = ["ENERGY", "GEO_OPT", "CELL_OPT", "MD", "VIBRATIONAL_ANALYSIS", "BAND", "EHRENFEST"]
        for i, rt in enumerate(run_types, 1):
            print(f"{i}. {rt}")
        
        rt_choice = input("Select run type (1-7) [1]: ").strip() or "1"
        try:
            idx = int(rt_choice) - 1
            cp2k_config['global']['RUN_TYPE'] = run_types[idx]
        except (ValueError, IndexError):
            print("Invalid selection. Using ENERGY.")
            cp2k_config['global']['RUN_TYPE'] = "ENERGY"
        
        print_level = input("Print level (LOW/MEDIUM/HIGH/DEBUG) [MEDIUM]: ").strip().upper() or "MEDIUM"
        cp2k_config['global']['PRINT_LEVEL'] = print_level
        
        # 2. DFT settings
        print("\n--- DFT Settings ---")
        
        # Exchange-correlation functional
        print("\nAvailable XC functionals:")
        functionals = ["PBE", "PBE0", "B3LYP", "BLYP", "BP86", "TPSS", "SCAN", "HSE06"]
        for i, func in enumerate(functionals, 1):
            print(f"{i}. {func}")
        
        func_choice = input("Select XC functional (1-8) [1]: ").strip() or "1"
        try:
            idx = int(func_choice) - 1
            functional = functionals[idx]
        except (ValueError, IndexError):
            print("Invalid selection. Using PBE.")
            functional = "PBE"
        
        # Handle hybrid functionals differently
        if functional in ["PBE0", "B3LYP", "HSE06"]:
            cp2k_config['dft']['xc']['XC_FUNCTIONAL'] = {functional: {}}
        else:
            cp2k_config['dft']['xc']['XC_FUNCTIONAL'] = functional
        
        # Basis set and potential files
        cp2k_config['dft']['BASIS_SET_FILE_NAME'] = "BASIS_MOLOPT"
        cp2k_config['dft']['POTENTIAL_FILE_NAME'] = "GTH_POTENTIALS"
        
        # vdW correction
        vdw = input("\nInclude van der Waals correction? (y/n) [n]: ").strip().lower() or "n"
        if vdw == "y":
            print("vdW correction methods:")
            print("1. D3 (Grimme's DFT-D3)")
            print("2. D3(BJ) (DFT-D3 with Becke-Johnson damping)")
            vdw_choice = input("Select method (1-2) [1]: ").strip() or "1"
            
            cp2k_config['dft']['xc']['VDW_POTENTIAL'] = {
                'POTENTIAL_TYPE': 'PAIR_POTENTIAL',
                'PAIR_POTENTIAL': {
                    'TYPE': 'DFTD3',
                    'PARAMETER_FILE_NAME': 'dftd3.dat',
                    'REFERENCE_FUNCTIONAL': functional
                }
            }
            
            if vdw_choice == "2":
                cp2k_config['dft']['xc']['VDW_POTENTIAL']['PAIR_POTENTIAL']['REFERENCE_FUNCTIONAL'] = functional + "-BJ"
        
        # 3. SCF settings
        print("\n--- SCF Settings ---")
        
        eps_scf = input("SCF convergence criterion (EPS_SCF) [1.0e-6]: ").strip() or "1.0e-6"
        try:
            cp2k_config['scf']['EPS_SCF'] = float(eps_scf)
        except ValueError:
            print("Invalid value. Using default 1.0e-6.")
            cp2k_config['scf']['EPS_SCF'] = 1.0e-6
        
        max_scf = input("Maximum number of SCF iterations (MAX_SCF) [50]: ").strip() or "50"
        try:
            cp2k_config['scf']['MAX_SCF'] = int(max_scf)
        except ValueError:
            print("Invalid value. Using default 50.")
            cp2k_config['scf']['MAX_SCF'] = 50
        
        cp2k_config['scf']['SCF_GUESS'] = "ATOMIC"
        
        # 4. MGRID settings (plane wave grid)
        print("\n--- Plane Wave Grid Settings ---")
        
        cutoff = input("Plane wave cutoff (CUTOFF) in Ry [400]: ").strip() or "400"
        try:
            cp2k_config['mgrid']['CUTOFF'] = int(cutoff)
        except ValueError:
            print("Invalid value. Using default 400.")
            cp2k_config['mgrid']['CUTOFF'] = 400
        
        rel_cutoff = input("Relative cutoff (REL_CUTOFF) in Ry [50]: ").strip() or "50"
        try:
            cp2k_config['mgrid']['REL_CUTOFF'] = int(rel_cutoff)
        except ValueError:
            print("Invalid value. Using default 50.")
            cp2k_config['mgrid']['REL_CUTOFF'] = 50
        
        ngrids = input("Number of multigrids (NGRIDS) [4]: ").strip() or "4"
        try:
            cp2k_config['mgrid']['NGRIDS'] = int(ngrids)
        except ValueError:
            print("Invalid value. Using default 4.")
            cp2k_config['mgrid']['NGRIDS'] = 4
        
        # 5. Run type specific settings
        run_type = cp2k_config['global']['RUN_TYPE']
        
        if run_type == "GEO_OPT":
            print("\n--- Geometry Optimization Settings ---")
            cp2k_config['motion'] = {'geo_opt': {}}
            
            optimizer = input("Optimizer (BFGS/CG/LBFGS) [BFGS]: ").strip().upper() or "BFGS"
            cp2k_config['motion']['geo_opt']['OPTIMIZER'] = optimizer
            
            max_iter = input("Maximum number of optimization steps [200]: ").strip() or "200"
            try:
                cp2k_config['motion']['geo_opt']['MAX_ITER'] = int(max_iter)
            except ValueError:
                cp2k_config['motion']['geo_opt']['MAX_ITER'] = 200
            
            max_force = input("Convergence criterion for forces (MAX_FORCE) in a.u. [0.00045]: ").strip() or "0.00045"
            try:
                cp2k_config['motion']['geo_opt']['MAX_FORCE'] = float(max_force)
            except ValueError:
                cp2k_config['motion']['geo_opt']['MAX_FORCE'] = 0.00045
        
        elif run_type == "CELL_OPT":
            print("\n--- Cell Optimization Settings ---")
            cp2k_config['motion'] = {'cell_opt': {}}
            
            optimizer = input("Optimizer (BFGS/CG/LBFGS) [BFGS]: ").strip().upper() or "BFGS"
            cp2k_config['motion']['cell_opt']['OPTIMIZER'] = optimizer
            
            max_iter = input("Maximum number of optimization steps [200]: ").strip() or "200"
            try:
                cp2k_config['motion']['cell_opt']['MAX_ITER'] = int(max_iter)
            except ValueError:
                cp2k_config['motion']['cell_opt']['MAX_ITER'] = 200
            
            pressure = input("Target pressure in bar [0.0]: ").strip() or "0.0"
            try:
                cp2k_config['motion']['cell_opt']['PRESSURE_TOLERANCE'] = float(pressure)
            except ValueError:
                cp2k_config['motion']['cell_opt']['PRESSURE_TOLERANCE'] = 0.0
            
            cp2k_config['motion']['cell_opt']['TYPE'] = "DIRECT_CELL_OPT"
        
        elif run_type == "MD":
            print("\n--- Molecular Dynamics Settings ---")
            cp2k_config['motion'] = {'md': {}}
            
            ensemble = input("Ensemble (NVE/NVT/NPT_I) [NVT]: ").strip().upper() or "NVT"
            cp2k_config['motion']['md']['ENSEMBLE'] = ensemble
            
            steps = input("Number of MD steps [1000]: ").strip() or "1000"
            try:
                cp2k_config['motion']['md']['STEPS'] = int(steps)
            except ValueError:
                cp2k_config['motion']['md']['STEPS'] = 1000
            
            timestep = input("Timestep in fs [0.5]: ").strip() or "0.5"
            try:
                cp2k_config['motion']['md']['TIMESTEP'] = float(timestep)
            except ValueError:
                cp2k_config['motion']['md']['TIMESTEP'] = 0.5
            
            temperature = input("Temperature in K [300.0]: ").strip() or "300.0"
            try:
                cp2k_config['motion']['md']['TEMPERATURE'] = float(temperature)
            except ValueError:
                cp2k_config['motion']['md']['TEMPERATURE'] = 300.0
            
            if ensemble in ["NVT", "NPT_I"]:
                thermostat = input("Thermostat (NOSE/CSVR) [NOSE]: ").strip().upper() or "NOSE"
                cp2k_config['motion']['md']['thermostat'] = {
                    'TYPE': thermostat,
                    'REGION': 'MASSIVE',
                    'TIMECON': 100.0
                }
        
        # 6. Cell settings
        print("\n--- Simulation Cell Settings ---")
        
        if 'cell' not in cp2k_config['subsys']:
            cp2k_config['subsys']['cell'] = {}
        
        # For single molecule calculations, ask about cell size
        if not cp2k_config['global']['RUN_TYPE'] in ["CELL_OPT"]:
            use_box = input("Set custom simulation box? (y/n) [n]: ").strip().lower() or "n"
            
            if use_box == "y":
                box_type = input("Box type (cubic/orthorhombic) [cubic]: ").strip().lower() or "cubic"
                
                if box_type == "cubic":
                    cell_size = input("Cell size (A) [15.0]: ").strip() or "15.0"
                    try:
                        size = float(cell_size)
                        cp2k_config['subsys']['cell']['ABC'] = f"{size} {size} {size}"
                    except ValueError:
                        print("Invalid value. Using default 15.0 Å.")
                        cp2k_config['subsys']['cell']['ABC'] = "15.0 15.0 15.0"
                else:
                    cell_a = input("Cell size a (A) [15.0]: ").strip() or "15.0"
                    cell_b = input("Cell size b (A) [15.0]: ").strip() or "15.0"
                    cell_c = input("Cell size c (A) [15.0]: ").strip() or "15.0"
                    
                    try:
                        a = float(cell_a)
                        b = float(cell_b)
                        c = float(cell_c)
                        cp2k_config['subsys']['cell']['ABC'] = f"{a} {b} {c}"
                    except ValueError:
                        print("Invalid values. Using default 15.0 Å for all dimensions.")
                        cp2k_config['subsys']['cell']['ABC'] = "15.0 15.0 15.0"
            else:
                cp2k_config['subsys']['cell']['ABC'] = "15.0 15.0 15.0"
        
        # 7. Element-specific settings
        print("\n--- Element-specific Settings ---")
        add_kinds = input("Specify basis sets for specific elements? (y/n) [n]: ").strip().lower() or "n"
        
        if add_kinds == "y":
            cp2k_config['kind_parameters'] = {}
            
            print("Available basis sets:")
            print("1. DZVP-MOLOPT-SR-GTH (Default, balanced)")
            print("2. TZV2P-MOLOPT-GTH (Triple-zeta, more accurate)")
            print("3. SZV-MOLOPT-SR-GTH (Single-zeta, faster)")
            
            while True:
                element = input("\nEnter element symbol (or press Enter to finish): ").strip()
                if not element:
                    break
                
                basis_set = input(f"Basis set for {element} (1/2/3) [1]: ").strip() or "1"
                basis_dict = {
                    "1": "DZVP-MOLOPT-SR-GTH",
                    "2": "TZV2P-MOLOPT-GTH",
                    "3": "SZV-MOLOPT-SR-GTH"
                }
                
                basis = basis_dict.get(basis_set, "DZVP-MOLOPT-SR-GTH")
                
                cp2k_config['kind_parameters'][element] = {
                    "BASIS_SET": basis,
                    "POTENTIAL": "GTH-PBE"
                }
                
                print(f"Added settings for {element}: {basis}")
        
        # 8. Add print options for forces
        print("\n--- Output and Printing Options ---")
        print_forces = input("Print forces for each SCF step? (y/n) [y]: ").strip().lower() or "y"
        
        if print_forces == "y":
            if 'print' not in cp2k_config:
                cp2k_config['print'] = {}
            
            cp2k_config['print']['forces'] = {
                'section_parameters': 'ON'
            }
            
            if 'scf' not in cp2k_config:
                cp2k_config['scf'] = {}
            
            if 'print' not in cp2k_config['scf']:
                cp2k_config['scf']['print'] = {}
            
            cp2k_config['scf']['print']['forces'] = {
                'section_parameters': 'ON',
                'each': {'qs_scf': 1}
            }
            
            # For specific run types, add more granular printing
            run_type = cp2k_config['global']['RUN_TYPE']
            if run_type == 'GEO_OPT':
                cp2k_config['print']['forces']['each'] = {'geo_opt': 1}
            elif run_type == 'MD':
                cp2k_config['print']['forces']['each'] = {'md': 1}
        
        # [existing code...]
        
        print("\nCP2K parameters configured successfully!")
        return cp2k_config

    def _get_cp2k_template(self):
        """Return a CP2K template configuration."""
        print("\nCP2K templates:\n1. GEO_OPT(PBE)\n2. ENERGY(PBE0)\n3. MD(PBE)\n4. VIBRATIONAL(PBE)")
        choice = input("Select (1-4): ").strip()
        templates = {
            "1": {
                "template_name": "Geometry optimization (PBE)",
                "global": {"RUN_TYPE": "GEO_OPT", "PRINT_LEVEL": "MEDIUM", "PROJECT": "cp2k_geo_opt"},
                "dft":    {"BASIS_SET_FILE_NAME": "BASIS_MOLOPT", "POTENTIAL_FILE_NAME": "GTH_POTENTIALS", "xc": {"XC_FUNCTIONAL": "PBE"}},
                "scf":    {"SCF_GUESS": "ATOMIC", "EPS_SCF": 1.0E-6, "MAX_SCF": 50, 
                        "print": {"forces": {"section_parameters": "ON", "each": {"qs_scf": 1}}}},
                "subsys": {"cell": {"ABC": "10.0 10.0 10.0"}},
                "print":  {"forces": {"section_parameters": "ON", "each": {"geo_opt": 1}}}
            },
            "2": {
                "template_name": "Electronic structure (PBE0)",
                "global": {"RUN_TYPE": "ENERGY", "PRINT_LEVEL": "MEDIUM", "PROJECT": "cp2k_energy"},
                "dft":    {"BASIS_SET_FILE_NAME": "BASIS_MOLOPT", "POTENTIAL_FILE_NAME": "GTH_POTENTIALS", "xc": {"XC_FUNCTIONAL": {"PBE0": {}}}},
                "scf":    {"SCF_GUESS": "ATOMIC", "EPS_SCF": 1.0E-6, "MAX_SCF": 100,
                        "print": {"forces": {"section_parameters": "ON", "each": {"qs_scf": 1}}}},
                "subsys": {"cell": {"ABC": "10.0 10.0 10.0"}},
                "print":  {"forces": {"section_parameters": "ON"}}
            },
            "3": {
                "template_name": "Molecular dynamics (PBE)",
                "global": {"RUN_TYPE": "MD", "PRINT_LEVEL": "LOW", "PROJECT": "cp2k_md"},
                "dft":    {"BASIS_SET_FILE_NAME": "BASIS_MOLOPT", "POTENTIAL_FILE_NAME": "GTH_POTENTIALS", "xc": {"XC_FUNCTIONAL": "PBE"}},
                "scf":    {"SCF_GUESS": "ATOMIC", "EPS_SCF": 1.0E-5, "MAX_SCF": 20,
                        "print": {"forces": {"section_parameters": "ON", "each": {"qs_scf": 1}}}},
                "motion": {"md": {"ENSEMBLE": "NVT", "STEPS": 1000, "TIMESTEP": 0.5, "TEMPERATURE": 300.0,
                                "thermostat": {"TYPE": "NOSE", "REGION": "MASSIVE", "TIMECON": 100.0}}},
                "subsys": {"cell": {"ABC": "10.0 10.0 10.0"}},
                "print":  {"forces": {"section_parameters": "ON", "each": {"md": 1}}}
            },
            "4": {
                "template_name": "Vibrational analysis (PBE)",
                "global": {"RUN_TYPE": "VIBRATIONAL_ANALYSIS", "PRINT_LEVEL": "MEDIUM", "PROJECT": "cp2k_vib"},
                "dft":    {"BASIS_SET_FILE_NAME": "BASIS_MOLOPT", "POTENTIAL_FILE_NAME": "GTH_POTENTIALS", "xc": {"XC_FUNCTIONAL": "PBE"}},
                "scf":    {"SCF_GUESS": "ATOMIC", "EPS_SCF": 1.0E-6, "MAX_SCF": 50,
                        "print": {"forces": {"section_parameters": "ON", "each": {"qs_scf": 1}}}},
                "vibrational_analysis": {"INTENSITIES": "TRUE", "THERMOCHEMISTRY": "TRUE", "DX": 0.01},
                "subsys": {"cell": {"ABC": "10.0 10.0 10.0"}},
                "print":  {"forces": {"section_parameters": "ON"}}
            },
        }
        return templates.get(choice, templates["1"])

    def _get_cp2k_custom_params(self):
        """Get custom parameters for CP2K."""
        params = {"global": {}, "dft": {"xc": {}}, "scf": {}, "subsys": {"cell": {}}, "print": {}}
        
        # [existing code...]
        
        # Add force printing
        print_forces = input("Print forces for each SCF step? (y/n) [y]: ").strip().lower() or "y"
        if print_forces == "y":
            params["print"]["forces"] = {"section_parameters": "ON"}
            
            if "scf" not in params:
                params["scf"] = {}
            if "print" not in params["scf"]:
                params["scf"]["print"] = {}
                
            params["scf"]["print"]["forces"] = {
                "section_parameters": "ON",
                "each": {"qs_scf": 1}
            }
            
            # Add run type specific printing
            if params["global"]["RUN_TYPE"] == "GEO_OPT":
                params["print"]["forces"]["each"] = {"geo_opt": 1}
            elif params["global"]["RUN_TYPE"] == "MD":
                params["print"]["forces"]["each"] = {"md": 1}
        
        return params



######## VASP INPUT GENERATION METHODS ##########

    def _handle_vasp_input_generation(self, use_supercell=False, supercell_dims=None):
        """Handle VASP input generation with detailed step-by-step guidance."""
        print("\n==== VASP Input File Generation (Guided Mode) ====\n")
        
        # Step 1: Get structure file path
        file_path = input("Enter the path to your structure file or directory (XYZ or CIF): ")
        input_path = Path(file_path).expanduser().resolve()
        
        # Validate the input file/directory
        if not input_path.exists():
            print(f"Error: Path '{input_path}' does not exist.")
            return
        
        # Step 2: Output location
        output_path = input("Enter the output directory path where input files will be generated: ")
        output_dir = Path(output_path).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Step 3: If directory - ask for batch processing or select specific file
        if input_path.is_dir():
            cif_files = list(input_path.glob("*.cif"))
            xyz_files = list(input_path.glob("*.xyz"))
            
            if not (cif_files or xyz_files):
                print(f"No .cif or .xyz files found in {input_path}")
                return
            
            print(f"\nFound {len(cif_files)} CIF files and {len(xyz_files)} XYZ files.")
            process_mode = input("Process all files or select a specific file? (all/select): ").strip().lower()
            
            if process_mode == "select":
                # Show files for selection
                all_files = cif_files + xyz_files
                for i, f in enumerate(all_files, 1):
                    print(f"{i}. {f.name}")
                
                try:
                    selected = int(input("\nSelect file number: "))
                    if 1 <= selected <= len(all_files):
                        input_path = all_files[selected-1]
                    else:
                        print("Invalid selection. Using first file.")
                        input_path = all_files[0]
                except ValueError:
                    print("Invalid input. Using first file.")
                    input_path = all_files[0]
                    
                print(f"\nSelected file: {input_path.name}")
                
                # Now input_path is a single file
                # Process with supercell if requested
                if use_supercell and supercell_dims:
                    self._process_supercell_vasp_file(input_path, output_dir, supercell_dims)
                else:
                    self._process_single_vasp_file(input_path, output_dir)
            else:
                # Proceed with batch processing
                print("\nBatch processing all structure files...\n")
                
                # Step 4: Get VASP parameters (once for all files)
                vasp_config, kpoints_config = self._get_vasp_parameters_interactively()
                
                # Process all files
                processed = 0
                
                if cif_files:
                    print(f"\nProcessing {len(cif_files)} CIF files...")
                    for cif_file in cif_files:
                        try:
                            print(f"  Processing {cif_file.name}...")
                            
                            # Create a subdirectory for this structure
                            struct_dir = output_dir / cif_file.stem
                            struct_dir.mkdir(parents=True, exist_ok=True)
                            
                            # Handle supercell creation if requested
                            if use_supercell and supercell_dims:
                                self._process_supercell_vasp_file(cif_file, struct_dir, supercell_dims, vasp_config, kpoints_config)
                            else:
                                # Original processing
                                xyz_path = output_dir / f"{cif_file.stem}.xyz"
                                if cif_to_xyz(cif_file, xyz_path):
                                    self._copy_to_xyz_checks(xyz_path, output_dir)
                                    
                                    # Generate VASP inputs - IMPORTANT: pass kpoints_config
                                    self._generate_vasp_input_for_file(xyz_path, struct_dir, vasp_config, kpoints_config)
                                    processed += 1
                            
                        except Exception as e:
                            print(f"  Error processing {cif_file.name}: {e}")
                
                if xyz_files:
                    print(f"\nProcessing {len(xyz_files)} XYZ files...")
                    for xyz_file in xyz_files:
                        try:
                            print(f"  Processing {xyz_file.name}...")
                            
                            # Create a subdirectory for this structure
                            struct_dir = output_dir / xyz_file.stem
                            struct_dir.mkdir(parents=True, exist_ok=True)
                            
                            # Handle supercell creation if requested
                            if use_supercell and supercell_dims:
                                self._process_supercell_vasp_file(xyz_file, struct_dir, supercell_dims, vasp_config, kpoints_config)
                            else:
                                # Original processing
                                self._copy_to_xyz_checks(xyz_file, output_dir)
                                
                                # Generate VASP inputs - IMPORTANT: pass kpoints_config
                                self._generate_vasp_input_for_file(xyz_file, struct_dir, vasp_config, kpoints_config)
                                processed += 1
                        except Exception as e:
                            print(f"  Error processing {xyz_file.name}: {e}")
                
                print(f"\nProcessed {processed} files. VASP input files generated in {output_dir}")
        else:
            # Process single file
            if use_supercell and supercell_dims:
                self._process_supercell_vasp_file(input_path, output_dir, supercell_dims)
            else:
                self._process_single_vasp_file(input_path, output_dir)

        # Also update the verification to include VASP directories
        self._verify_vasp_output_files(output_dir)

    def _verify_vasp_output_files(self, output_dir):
        """Verify that VASP input files are present in the output directory."""
        output_dir = Path(output_dir)
        
        # For VASP, we expect subdirectories containing the input files
        vasp_dirs = [d for d in output_dir.iterdir() if d.is_dir() and not d.name == "xyz_checks" and not d.name == "temp_supercell"]
        
        vasp_files = []
        for d in vasp_dirs:
            incar_files = list(d.glob("INCAR"))
            poscar_files = list(d.glob("POSCAR"))
            kpoints_files = list(d.glob("KPOINTS"))
            if incar_files and poscar_files and kpoints_files:
                vasp_files.append(d)
        
        if not vasp_files:
            print("WARNING: No complete VASP input sets found in output directory! Something may have gone wrong.")
            print(f"Output directory contents: {[f.name for f in output_dir.iterdir()]}")
        else:
            print(f"Verification successful: {len(vasp_files)} VASP input sets found in output directory.")

    def _generate_vasp_input_for_file(self, xyz_file, output_dir, vasp_config, kpoints_config=None):
        """Generate VASP input files for a single XYZ file."""
        from multi_agent_dft.file_processing.xyz import parse_xyz
        from pathlib import Path
        import os
            
        try:
            # Parse XYZ file
            structure_data = parse_xyz(xyz_file)
            if not structure_data:
                print(f"Error: Could not parse XYZ file {xyz_file}")
                return False
            
            # If kpoints_config is not provided, extract it from vasp_config
            if kpoints_config is None and 'KPOINTS' in vasp_config:
                kpoints_config = vasp_config.pop('KPOINTS')
            elif kpoints_config is None:
                # Default kpoints configuration
                kpoints_config = {'type': 'gamma', 'grid': [3, 3, 3]}
            
            # Make sure output_dir is a Path object
            output_dir = Path(output_dir)
            
            # Create the output directory if it doesn't exist
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate INCAR file
            incar_path = output_dir / "INCAR"
            with open(incar_path, "w") as f:
                f.write(self._generate_vasp_incar(vasp_config))
            
            # Generate POSCAR file
            poscar_path = output_dir / "POSCAR"
            with open(poscar_path, "w") as f:
                f.write(self._generate_vasp_poscar(structure_data))
            
            # Generate KPOINTS file
            kpoints_path = output_dir / "KPOINTS"
            with open(kpoints_path, "w") as f:
                f.write(self._generate_vasp_kpoints(kpoints_config))
            
            print(f"  Generated VASP input files in: {output_dir}")
            return True
        except Exception as e:
            print(f"  Error generating VASP input for {xyz_file}: {e}")
            return False

    def _process_single_vasp_file(self, file_path, output_dir):
        """Process a single file for VASP input generation."""
        # Validate the file
        valid, msg, _ = validate_structure(file_path)
        if not valid:
            print(f"Structure validation failed: {msg}")
            return
        
        # For CIF files, convert to XYZ first
        if file_path.suffix.lower() == '.cif':
            print(f"\nConverting CIF file to XYZ format...")
            xyz_path = output_dir / f"{file_path.stem}.xyz"
            if not cif_to_xyz(file_path, xyz_path):
                print("CIF conversion failed. Aborting.")
                return
            file_path = xyz_path
            print(f"Conversion successful: {xyz_path}")
            self._copy_to_xyz_checks(xyz_path, output_dir)
        
        # Get VASP parameters through interactive prompts
        vasp_config, kpoints_config = self._get_vasp_parameters_interactively()
        
        # Create a subdirectory for this structure
        struct_dir = output_dir / file_path.stem
        struct_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate VASP input files
        self._generate_vasp_input_for_file(file_path, struct_dir, vasp_config, kpoints_config)
        
        print(f"\nVASP input files generated in {struct_dir}")

    def _process_supercell_vasp_file(self, file_path, output_dir, supercell_dims, vasp_config=None, kpoints_config=None):
        """Process a single file with supercell creation for VASP input generation."""
        nx, ny, nz = supercell_dims
        
        # Validate the file
        valid, msg, _ = validate_structure(file_path)
        if not valid:
            print(f"Structure validation failed: {msg}")
            return
        
        # Create a temporary directory for supercell processing
        temp_dir = output_dir / "temp_supercell"
        temp_dir.mkdir(exist_ok=True)
        
        try:
            from ase.io import read, write
            from ase.build import make_supercell
            import numpy as np
            
            # For CIF files, convert to XYZ first if needed
            if file_path.suffix.lower() == ".cif":
                print(f"\nConverting CIF file to XYZ format...")
                xyz_path = temp_dir / f"{file_path.stem}.xyz"
                if not cif_to_xyz(file_path, xyz_path):
                    print("CIF conversion failed. Aborting.")
                    return
                file_path = xyz_path
                print(f"Conversion successful: {xyz_path}")
                self._copy_to_xyz_checks(xyz_path, output_dir)
            
            # Create the supercell
            print(f"Creating {nx}x{ny}x{nz} supercell...")
            ase_struct = read(file_path)
            NxNxN = np.array([nx, ny, nz])
            supercell = make_supercell(ase_struct, np.eye(3) * NxNxN)
            
            # Save the supercell to a temporary file
            supercell_file = temp_dir / f"{file_path.stem}_{nx}x{ny}x{nz}.xyz"
            write(supercell_file, supercell)
            print(f"Supercell created with {len(supercell)} atoms")
            
            # Get VASP parameters if not provided
            if vasp_config is None or kpoints_config is None:
                vasp_config, kpoints_config = self._get_vasp_parameters_interactively()
            
            # Generate VASP input for the supercell
            self._generate_vasp_input_for_file(supercell_file, output_dir, vasp_config, kpoints_config)
            
        except Exception as e:
            print(f"Error in supercell creation or VASP input generation: {e}")
        finally:
            # Clean up temporary directory
            import shutil
            shutil.rmtree(temp_dir)

    def _get_vasp_parameters_interactively(self):
        """Interactive questionnaire for VASP input parameters."""
        print("\n=== VASP Calculation Settings ===")
        
        # Initialize configs
        vasp_config = {}
        kpoints_config = {}
        
        try:
            # 1. Basic Settings
            print("\n--- Basic Settings ---")
            system_name = input("System description [VASP calculation]: ").strip() or "VASP calculation"
            vasp_config['SYSTEM'] = system_name
            
            print("\nAvailable calculation types:")
            calc_types = [
                "Static (single-point energy)",
                "Structure Optimization",
                "Cell Optimization",
                "Molecular Dynamics",
                "Band Structure"
            ]
            for i, calc_type in enumerate(calc_types, 1):
                print(f"{i}. {calc_type}")
            
            calc_choice = input("Select calculation type (1-5) [2]: ").strip() or "2"
            try:
                idx = int(calc_choice) - 1
                calc_type = calc_types[idx]
            except (ValueError, IndexError):
                print("Invalid selection. Using Structure Optimization.")
                calc_type = "Structure Optimization"
            
            # 2. Configure parameters based on calculation type
            if "Static" in calc_type:
                vasp_config['IBRION'] = -1
                vasp_config['NSW'] = 0
                vasp_config['ISIF'] = 2
            elif "Structure Optimization" in calc_type:
                vasp_config['IBRION'] = 2
                vasp_config['NSW'] = 100
                vasp_config['ISIF'] = 2
                
                # Ask for more detailed optimization settings
                print("\n--- Optimization Settings ---")
                
                optimizer = input("Optimizer (1=Conjugate Gradient, 3=Damped MD) [2]: ").strip() or "2"
                try:
                    vasp_config['IBRION'] = int(optimizer)
                except ValueError:
                    print("Invalid input. Using Conjugate Gradient (IBRION=2).")
                    vasp_config['IBRION'] = 2
                
                steps = input("Maximum number of optimization steps [100]: ").strip() or "100"
                try:
                    vasp_config['NSW'] = int(steps)
                except ValueError:
                    print("Invalid input. Using 100 steps.")
                    vasp_config['NSW'] = 100
                
                force_conv = input("Force convergence criterion in eV/Å [0.01]: ").strip() or "0.01"
                try:
                    vasp_config['EDIFFG'] = -1 * abs(float(force_conv))  # Negative for force-based convergence
                except ValueError:
                    print("Invalid input. Using 0.01 eV/Å.")
                    vasp_config['EDIFFG'] = -0.01
                    
            elif "Cell Optimization" in calc_type:
                vasp_config['IBRION'] = 2
                vasp_config['NSW'] = 100
                vasp_config['ISIF'] = 3  # Full cell and ionic relaxation
                
                # Ask for more detailed optimization settings
                print("\n--- Cell Optimization Settings ---")
                
                isif = input("ISIF (3=full cell+ions, 4=fixed cell shape, 7=volume only) [3]: ").strip() or "3"
                try:
                    vasp_config['ISIF'] = int(isif)
                except ValueError:
                    print("Invalid input. Using full cell and ionic relaxation (ISIF=3).")
                    vasp_config['ISIF'] = 3
                
                steps = input("Maximum number of optimization steps [100]: ").strip() or "100"
                try:
                    vasp_config['NSW'] = int(steps)
                except ValueError:
                    print("Invalid input. Using 100 steps.")
                    vasp_config['NSW'] = 100
                    
            elif "Molecular Dynamics" in calc_type:
                vasp_config['IBRION'] = 0
                vasp_config['NSW'] = 1000
                vasp_config['ISIF'] = 2
                vasp_config['TEBEG'] = 300
                vasp_config['SMASS'] = 0
                
                # Ask for more detailed MD settings
                print("\n--- Molecular Dynamics Settings ---")
                
                steps = input("Number of MD steps [1000]: ").strip() or "1000"
                try:
                    vasp_config['NSW'] = int(steps)
                except ValueError:
                    print("Invalid input. Using 1000 steps.")
                    vasp_config['NSW'] = 1000
                
                temp = input("Temperature in K [300]: ").strip() or "300"
                try:
                    vasp_config['TEBEG'] = float(temp)
                    vasp_config['TEEND'] = float(temp)  # Constant temperature
                except ValueError:
                    print("Invalid input. Using 300K.")
                    vasp_config['TEBEG'] = 300
                    vasp_config['TEEND'] = 300
                
                time_step = input("Time step in fs [1.0]: ").strip() or "1.0"
                try:
                    vasp_config['POTIM'] = float(time_step)
                except ValueError:
                    print("Invalid input. Using 1.0 fs.")
                    vasp_config['POTIM'] = 1.0
                    
            elif "Band Structure" in calc_type:
                vasp_config['IBRION'] = -1
                vasp_config['NSW'] = 0
                vasp_config['ISIF'] = 2
                vasp_config['ICHARG'] = 11  # Non-selfconsistent using CHGCAR
                vasp_config['LORBIT'] = 11  # Output PROCAR for band analysis
                
                # Set up line-mode for k-points
                kpoints_config['type'] = 'line'
                kpoints_config['line_mode'] = True
            
            # 3. Electronic Structure Settings
            print("\n--- Electronic Structure Settings ---")
            
            # Exchange-correlation functional
            print("\nAvailable exchange-correlation functionals:")
            functionals = ["PBE", "PBE-Sol", "LDA", "revPBE", "SCAN", "BEEF-vdW", "vdW-DF", "vdW-DF2", "optPBE-vdW"]
            for i, func in enumerate(functionals, 1):
                print(f"{i}. {func}")
            
            func_choice = input("Select XC functional (1-9) [1]: ").strip() or "1"
            try:
                idx = int(func_choice) - 1
                functional = functionals[idx]
            except (ValueError, IndexError):
                print("Invalid selection. Using PBE.")
                functional = "PBE"
            
            # Set parameters based on functional choice
            if functional == "PBE":
                vasp_config['GGA'] = "PE"
            elif functional == "PBE-Sol":
                vasp_config['GGA'] = "PS"
            elif functional == "LDA":
                # LDA is the default, no need to set
                pass
            elif functional == "revPBE":
                vasp_config['GGA'] = "RE"
            elif functional == "SCAN":
                vasp_config['METAGGA'] = "SCAN"
            elif "vdW" in functional:
                if functional == "BEEF-vdW":
                    vasp_config['GGA'] = "BF"
                elif functional == "optPBE-vdW":
                    vasp_config['GGA'] = "OR"
                    vasp_config['LUSE_VDW'] = True
                    vasp_config['AGGAC'] = 0.0
                elif functional == "vdW-DF":
                    vasp_config['GGA'] = "RE"
                    vasp_config['LUSE_VDW'] = True
                    vasp_config['AGGAC'] = 0.0
                elif functional == "vdW-DF2":
                    vasp_config['GGA'] = "ML"
                    vasp_config['LUSE_VDW'] = True
                    vasp_config['AGGAC'] = 0.0
            
            # LREAL settings - FIXED INDENTATION
            lreal_options = input("\nSet LREAL for projector augmentation? (y/n) [n]: ").strip().lower() or "n"
            if lreal_options == "y":
                print("\nLREAL options:")
                print("1. Auto (recommended for systems >20-30 atoms)")
                print("2. On (faster but less accurate)")
                print("3. True (alias for On)")
                print("4. False (more accurate but slower, good for small cells)")
                
                lreal_choice = input("Select LREAL option (1-4) [1]: ").strip() or "1"
                if lreal_choice == "2":
                    vasp_config['LREAL'] = ".TRUE."
                elif lreal_choice == "3":
                    vasp_config['LREAL'] = "T"
                elif lreal_choice == "4":
                    vasp_config['LREAL'] = ".FALSE."
                else:
                    vasp_config['LREAL'] = "Auto"
            
            # vdW dispersion correction - FIXED INDENTATION
            disp_corr = input("\nAdd van der Waals (vdW) dispersion correction? (y/n) [n]: ").strip().lower() or "n"
            if disp_corr == "y":
                # If vdW-functional is already selected, confirm that user wants to add additional correction
                if vasp_config.get('LUSE_VDW', False):
                    print("\nWARNING: You've selected a vdW-aware functional that already includes dispersion effects.")
                    confirm = input("Still add empirical dispersion correction? (y/n) [n]: ").strip().lower() or "n"
                    if confirm != "y":
                        print("Skipping additional dispersion correction.")
                else:
                    # If we're here, either user confirmed or no vdW functional was selected
                    print("\nAvailable dispersion correction methods:")
                    print("1. DFT-D2 (Grimme's original method)")
                    print("2. DFT-D3 (Grimme's improved method)")
                    print("3. DFT-D3(BJ) (Grimme's method with Becke-Johnson damping)")
                    print("4. DFT-D4 (Grimme's latest method)")
                    print("5. TS (Tkatchenko-Scheffler method)")
                    print("6. MBD (Many-Body Dispersion method)")
                    
                    disp_method = input("Select dispersion method (1-5) [2]: ").strip() or "2"
                    
                    # Set appropriate VASP tags based on the selected method
                    if disp_method == "1":
                        # DFT-D2
                        vasp_config['LVDW'] = True
                        vasp_config['IVDW'] = 1
                        print("Added DFT-D2 dispersion correction")
                    elif disp_method == "3":
                        # DFT-D3(BJ)
                        vasp_config['LVDW'] = True
                        vasp_config['IVDW'] = 12
                        print("Added DFT-D3(BJ) dispersion correction")
                    elif disp_method == "4":
                        # TS method
                        vasp_config['LVDW'] = True
                        vasp_config['IVDW'] = 13
                        print("Added DFT-D4 dispersion correction")
                    elif disp_method == "5":
                        # TS method
                        vasp_config['LVDW'] = True
                        vasp_config['IVDW'] = 2
                        print("Added TS dispersion correction")
                    elif disp_method == "6":
                        # MBD method
                        vasp_config['LVDW'] = True
                        vasp_config['IVDW'] = 202
                        print("Added MBD dispersion correction")
                    else:
                        # Default to DFT-D3
                        vasp_config['LVDW'] = True
                        vasp_config['IVDW'] = 11
                        print("Added DFT-D3 dispersion correction")
                        
                    # Optional scaling factor for dispersion
                    scale = input("Dispersion scaling factor (optional, press Enter for default): ").strip()
                    if scale:
                        try:
                            vasp_config['VDW_S6'] = float(scale)
                        except ValueError:
                            print("Invalid scaling factor. Using default.")
                            
                    # Cutoff radius for dispersion calculation
                    cutoff = input("Dispersion cutoff radius in Angstrom (optional, press Enter for default): ").strip()
                    if cutoff:
                        try:
                            vasp_config['VDW_R0'] = float(cutoff)
                        except ValueError:
                            print("Invalid cutoff radius. Using default.")
            
            # Print LREAL info if set
            if 'LREAL' in vasp_config:
                print(f"LREAL: {vasp_config['LREAL']}")
                
            # Print dispersion info if set
            if vasp_config.get('LVDW', False):
                ivdw = vasp_config.get('IVDW', 0)
                if ivdw == 1:
                    print(f"Dispersion: DFT-D2")
                elif ivdw == 2:
                    print(f"Dispersion: TS method")
                elif ivdw == 11:
                    print(f"Dispersion: DFT-D3")
                elif ivdw == 12:
                    print(f"Dispersion: DFT-D3(BJ)")
                elif ivdw == 202:
                    print(f"Dispersion: MBD method")
                else:
                    print(f"Dispersion: IVDW={ivdw}")
                    
                if 'VDW_S6' in vasp_config:
                    print(f"Dispersion Scale: {vasp_config['VDW_S6']}")
                if 'VDW_R0' in vasp_config:
                    print(f"Dispersion Cutoff: {vasp_config['VDW_R0']} Å")
            
            # Plane wave cutoff - FIXED INDENTATION
            encut = input("\nPlane wave cutoff (ENCUT) in eV [400]: ").strip() or "400"
            try:
                vasp_config['ENCUT'] = float(encut)
            except ValueError:
                print("Invalid value. Using default 400 eV.")
                vasp_config['ENCUT'] = 400
            
            # SCF convergence
            ediff = input("Electronic convergence criterion (EDIFF) [1.0e-5]: ").strip() or "1.0e-5"
            try:
                vasp_config['EDIFF'] = float(ediff)
            except ValueError:
                print("Invalid value. Using default 1.0e-5.")
                vasp_config['EDIFF'] = 1.0e-5
            
            # Max SCF iterations
            nelm = input("Maximum number of SCF iterations (NELM) [60]: ").strip() or "60"
            try:
                vasp_config['NELM'] = int(nelm)
            except ValueError:
                print("Invalid value. Using default 60.")
                vasp_config['NELM'] = 60
            
            # 4. Electronic smearing - FIXED INDENTATION
            print("\n--- Electronic Smearing Settings ---")
            
            print("\nAvailable smearing methods:")
            smear_methods = [
                "0: Gaussian", 
                "-1: Fermi", 
                "1: Methfessel-Paxton (order 1)", 
                "2: Methfessel-Paxton (order 2)",
                "-5: Tetrahedron with Blöchl corrections"
            ]
            for method in smear_methods:
                print(method)
            
            ismear = input("Select smearing method (default 0): ").strip() or "0"
            try:
                vasp_config['ISMEAR'] = int(ismear)
            except ValueError:
                print("Invalid value. Using default 0 (Gaussian).")
                vasp_config['ISMEAR'] = 0
            
            # Only ask for sigma if not using tetrahedron method
            if vasp_config['ISMEAR'] != -5:
                sigma = input("Smearing width (SIGMA) in eV [0.05]: ").strip() or "0.05"
                try:
                    vasp_config['SIGMA'] = float(sigma)
                except ValueError:
                    print("Invalid value. Using default 0.05 eV.")
                    vasp_config['SIGMA'] = 0.05
            
            # 5. K-point Sampling - FIXED INDENTATION
            print("\n--- K-point Sampling Settings ---")
            
            if not kpoints_config.get('line_mode'):  # Skip if already set for band structure
                print("\nK-point generation methods:")
                print("1. Automatic (Gamma-centered)")
                print("2. Automatic (Monkhorst-Pack)")
                print("3. Explicitly specify k-points")
                
                kpoint_method = input("Select method (1-3) [1]: ").strip() or "1"
                
                if kpoint_method == "1":
                    kpoints_config['type'] = 'gamma'
                    grid = input("K-point grid (three integers separated by spaces) [3 3 3]: ").strip() or "3 3 3"
                    try:
                        kpoints_config['grid'] = [int(x) for x in grid.split()]
                    except ValueError:
                        print("Invalid grid. Using default 3x3x3.")
                        kpoints_config['grid'] = [3, 3, 3]
                        
                elif kpoint_method == "2":
                    kpoints_config['type'] = 'monkhorst'
                    grid = input("K-point grid (three integers separated by spaces) [3 3 3]: ").strip() or "3 3 3"
                    try:
                        kpoints_config['grid'] = [int(x) for x in grid.split()]
                    except ValueError:
                        print("Invalid grid. Using default 3x3x3.")
                        kpoints_config['grid'] = [3, 3, 3]
                    
                    shift = input("Grid shift (three values 0 or 0.5, separated by spaces) [0 0 0]: ").strip() or "0 0 0"
                    try:
                        kpoints_config['shift'] = [float(x) for x in shift.split()]
                    except ValueError:
                        print("Invalid shift. Using default 0 0 0.")
                        kpoints_config['shift'] = [0, 0, 0]
                        
                elif kpoint_method == "3":
                    kpoints_config['type'] = 'explicit'
                    print("\n--- Manual K-point Entry ---")
                    print("Enter k-points one by one. Format: kx ky kz weight")
                    print("Example: 0.0 0.0 0.0 1.0")
                    print("Enter a blank line when done.")
                    
                    kpoints_list = []
                    k_count = 0
                    while True:
                        k_input = input(f"K-point {k_count+1} (or press Enter to finish): ").strip()
                        if not k_input:
                            break
                            
                        try:
                            kx, ky, kz, weight = map(float, k_input.split())
                            kpoints_list.append([kx, ky, kz, weight])
                            k_count += 1
                        except ValueError:
                            print("Invalid format. Please use: kx ky kz weight")
                            
                    if kpoints_list:
                        kpoints_config['explicit_points'] = kpoints_list
                        kpoints_config['num_kpoints'] = len(kpoints_list)
                    else:
                        print("No valid k-points entered. Using gamma-centered 3x3x3 grid instead.")
                        kpoints_config['type'] = 'gamma'
                        kpoints_config['grid'] = [3, 3, 3]
                else:
                    # Default to gamma if invalid choice
                    print("Invalid choice. Using gamma-centered grid.")
                    kpoints_config['type'] = 'gamma'
                    kpoints_config['grid'] = [3, 3, 3]
            
            # 6. Output Settings - FIXED INDENTATION
            print("\n--- Output Settings ---")
            
            lwave = input("Write WAVECAR file? (y/n) [n]: ").strip().lower() or "n"
            vasp_config['LWAVE'] = (lwave == "y")
            
            lcharg = input("Write CHGCAR file? (y/n) [y]: ").strip().lower() or "y"
            vasp_config['LCHARG'] = (lcharg == "y")
            
            # For band structure, LORBIT controls projected band structure
            if "Band Structure" in calc_type:
                lorbit = input("Write PROCAR with orbital projections? (y/n) [y]: ").strip().lower() or "y"
                if lorbit == "y":
                    vasp_config['LORBIT'] = 11
            
            # 7. Advanced Settings - FIXED INDENTATION
            print("\n--- Advanced Settings ---")
            use_advanced = input("Configure advanced settings? (y/n) [n]: ").strip().lower() or "n"
            
            if use_advanced == "y":
                # Precision
                print("\nPrecision levels:")
                print("1. Normal")
                print("2. Accurate")
                print("3. High")
                
                prec = input("Select precision level (1-3) [2]: ").strip() or "2"
                if prec == "1":
                    vasp_config['PREC'] = "Normal"
                elif prec == "3":
                    vasp_config['PREC'] = "High"
                else:
                    vasp_config['PREC'] = "Accurate"
                
                # Electronic minimization algorithm
                print("\nElectronic minimization algorithms:")
                print("1. Normal (blocked Davidson)")
                print("2. Very Fast (RMM-DIIS)")
                print("3. Conjugate Gradient")
                print("4. All (best of Normal and Very Fast)")
                
                algo = input("Select algorithm (1-4) [1]: ").strip() or "1"
                if algo == "2":
                    vasp_config['ALGO'] = "VeryFast"
                elif algo == "3":
                    vasp_config['ALGO'] = "Conjugate"
                elif algo == "4":
                    vasp_config['ALGO'] = "All"
                else:
                    vasp_config['ALGO'] = "Normal"
                
                # Magnetic calculation
                mag_calc = input("\nPerform spin-polarized calculation? (y/n) [n]: ").strip().lower() or "n"
                if mag_calc == "y":
                    vasp_config['ISPIN'] = 2
                    
                    initial_mag = input("Initial magnetic moment per atom (or press Enter to skip): ").strip()
                    if initial_mag:
                        try:
                            vasp_config['MAGMOM'] = float(initial_mag)
                        except ValueError:
                            print("Invalid value. Not setting initial magnetic moment.")
            
            # Make sure kpoints_config has some default values if not set anywhere
            if not kpoints_config:
                kpoints_config = {'type': 'gamma', 'grid': [3, 3, 3]}
            
            # Summary of selected options - FIXED INDENTATION
            print("\n=== Summary of VASP Calculation Settings ===")
            print(f"System Description: {vasp_config['SYSTEM']}")
            print(f"Calculation Type:   {calc_type}")
            
            if "Optimization" in calc_type:
                print(f"Optimizer:         IBRION={vasp_config.get('IBRION', 'N/A')}")
                print(f"Max Steps:         NSW={vasp_config.get('NSW', 'N/A')}")
                if 'EDIFFG' in vasp_config:
                    print(f"Force Convergence: {abs(vasp_config['EDIFFG'])} eV/Å")
            
            print(f"XC Functional:      {functional}")
            if 'ENCUT' in vasp_config:
                print(f"Plane Wave Cutoff:  {vasp_config['ENCUT']} eV")
            if 'EDIFF' in vasp_config:
                print(f"Electronic Conv.:   {vasp_config['EDIFF']}")
            
            if kpoints_config.get('type') == 'gamma' and 'grid' in kpoints_config:
                grid = kpoints_config['grid']
                print(f"K-point Grid:       {grid[0]}x{grid[1]}x{grid[2]} (Gamma)")
            elif kpoints_config.get('type') == 'monkhorst' and 'grid' in kpoints_config:
                grid = kpoints_config['grid']
                print(f"K-point Grid:       {grid[0]}x{grid[1]}x{grid[2]} (Monkhorst-Pack)")
            elif kpoints_config.get('line_mode'):
                print(f"K-point Mode:       Line mode for band structure")
            
            print(f"Write WAVECAR:      {vasp_config.get('LWAVE', False)}")
            print(f"Write CHGCAR:       {vasp_config.get('LCHARG', True)}")
            
            if vasp_config.get('ISPIN') == 2:
                print(f"Spin Polarized:     Yes")
                if 'MAGMOM' in vasp_config:
                    print(f"Initial Mag. Mom.:  {vasp_config['MAGMOM']}")
            
            # Add dispersion correction to summary
            if vasp_config.get('LVDW', False):
                ivdw = vasp_config.get('IVDW', 0)
                if ivdw == 1:
                    print(f"Dispersion:        DFT-D2")
                elif ivdw == 2:
                    print(f"Dispersion:        TS method")
                elif ivdw == 11:
                    print(f"Dispersion:        DFT-D3")
                elif ivdw == 12:
                    print(f"Dispersion:        DFT-D3(BJ)")
                else:
                    print(f"Dispersion:        IVDW={ivdw}")
                
                if 'VDW_S6' in vasp_config:
                    print(f"Dispersion Scale:  {vasp_config['VDW_S6']}")
                if 'VDW_R0' in vasp_config:
                    print(f"Dispersion Cutoff: {vasp_config['VDW_R0']} Å")
            
            confirm = input("\nConfirm these settings? (y/n) [y]: ").strip().lower() or "y"
            if confirm != "y":
                print("Restarting parameter selection...")
                return self._get_vasp_parameters_interactively()
            
            # FIXED: This return statement was inside nested indentation blocks
            return vasp_config, kpoints_config

        except Exception as e:
            print(f"\n*** Error during VASP parameter setup: {e} ***")
            print("Using default parameters instead.")
            
            # Return default values in case of error
            default_vasp = {
                "SYSTEM": "Default VASP calculation",
                "IBRION": 2, "NSW": 100, "ISIF": 3,
                "ENCUT": 400, "EDIFF": 1e-5, "EDIFFG": -0.01,
                "ISMEAR": 0, "SIGMA": 0.05,
                "LWAVE": False, "LCHARG": True,
                "GGA": "PE"  # PBE functional
            }
            default_kpoints = {'type': 'gamma', 'grid': [3, 3, 3]}
            
            return default_vasp, default_kpoints

    def _get_vasp_template(self):
        """Return a VASP template configuration."""
        print("\nVASP templates:\n1. GEO_OPT\n2. STATIC\n3. BAND\n4. MD\n5. GEO_OPT with DFT-D3\n6. GEO_OPT with MBD")
        c = input("Select (1-6): ").strip()
        templates = {
            "1": {"SYSTEM":"Geometry optimization","ISTART":0,"ICHARG":2,"ENCUT":400,"ISMEAR":0,"SIGMA":0.05,
                "IBRION":2,"NSW":100,"ISIF":3,"EDIFF":1e-5,"EDIFFG":-0.01,"PREC":"Accurate","LWAVE":False,
                "LCHARG":True,"ALGO":"Normal","LREAL":"Auto","KPOINTS":{"type":"gamma","grid":[3,3,3]}},
            "2": {"SYSTEM":"Static calculation","ISTART":0,"ICHARG":2,"ENCUT":500,"ISMEAR":0,"SIGMA":0.05,
                "IBRION":-1,"NSW":0,"ISIF":2,"EDIFF":1e-6,"PREC":"Accurate","LWAVE":True,"LCHARG":True,
                "ALGO":"Normal","LREAL":"Auto","KPOINTS":{"type":"monkhorst","grid":[5,5,5]}},
            "3": {"SYSTEM":"Band structure","ISTART":0,"ICHARG":11,"ENCUT":450,"ISMEAR":0,"SIGMA":0.05,
                "IBRION":-1,"NSW":0,"ISIF":2,"EDIFF":1e-6,"PREC":"Accurate","LWAVE":True,"LCHARG":False,
                "ALGO":"Normal","LORBIT":11,"LREAL":"Auto","KPOINTS":{"type":"line","line_mode":True}},
            "4": {"SYSTEM":"Molecular dynamics","ISTART":0,"ICHARG":2,"ENCUT":350,"ISMEAR":0,"SIGMA":0.05,
                "IBRION":0,"NSW":1000,"ISIF":2,"EDIFF":1e-5,"PREC":"Normal","LWAVE":False,"LCHARG":False,
                "ALGO":"VeryFast","TEBEG":300,"TEEND":300,"SMASS":0,"POTIM":1,"LREAL":"Auto",
                "KPOINTS":{"type":"gamma","grid":[1,1,1]}},
            "5": {"SYSTEM":"Geometry optimization with DFT-D3","ISTART":0,"ICHARG":2,"ENCUT":400,"ISMEAR":0,"SIGMA":0.05,
                "IBRION":2,"NSW":100,"ISIF":3,"EDIFF":1e-5,"EDIFFG":-0.01,"PREC":"Accurate","LWAVE":False,
                "LCHARG":True,"ALGO":"Normal","LREAL":"Auto","LVDW":True,"IVDW":11,"KPOINTS":{"type":"gamma","grid":[3,3,3]}},
            "6": {"SYSTEM":"Geometry optimization with MBD","ISTART":0,"ICHARG":2,"ENCUT":400,"ISMEAR":0,"SIGMA":0.05,
                "IBRION":2,"NSW":100,"ISIF":3,"EDIFF":1e-5,"EDIFFG":-0.01,"PREC":"Accurate","LWAVE":False,
                "LCHARG":True,"ALGO":"Normal","LREAL":"Auto","LVDW":True,"IVDW":202,"KPOINTS":{"type":"gamma","grid":[3,3,3]}}
        }
        return templates.get(c,templates["1"])
    
    def _batch_input_generation(self, use_supercell=False, supercell_dims=None):
        """Auto‐convert all supported files in folder using default templates."""
        code = input("Batch‐mode: which DFT code? (CP2K/VASP/Gaussian): ").strip().lower()
        path = Path(input("Path to file or directory: ").strip()).expanduser().resolve()
        out  = Path(input("Output directory: ").strip()).expanduser().resolve()
        out.mkdir(parents=True, exist_ok=True)

        valid,msg,_ = validate_structure(path)
        if not valid:
            print(f"Validation failed: {msg}")
            return

        # pick default template
        if code=="cp2k":
            params = self._get_cp2k_template()
            fmt    = "cp2k";  ext=".inp"

        elif code == "vasp":
            params = self._get_vasp_template()
            kpoints = params.pop('KPOINTS', {'type': 'gamma', 'grid': [3, 3, 3]})  # Extract KPOINTS
            fmt = "vasp"
            ext = ""
            
        else:
            params = self._get_gaussian_params()
            fmt    = "gaussian"; ext=".com"

        files = path.glob("*.cif") if path.is_dir() else [path]
        if path.is_dir():
            files = list(path.glob("*.cif")) + list(path.glob("*.xyz"))

        for f in files:
            stem = f.stem.replace(" ","_")
            outp = out/(stem+ext)
            
            # Handle supercell creation if requested
            if use_supercell and supercell_dims:
                # Create a temporary directory for supercell processing
                temp_dir = out / "temp_supercell"
                temp_dir.mkdir(exist_ok=True)
                
                # Create supercell using ASE
                nx, ny, nz = supercell_dims
                try:
                    from ase.io import read, write
                    from ase.build import make_supercell
                    import numpy as np
                    
                    print(f"Creating {nx}x{ny}x{nz} supercell for {f.name}...")
                    ase_struct = read(f)
                    NxNxN = np.array([nx, ny, nz])
                    supercell = make_supercell(ase_struct, np.eye(3) * NxNxN)
                    
                    # Save the supercell to a temporary file
                    temp_file = temp_dir / f"{stem}_supercell.cif"
                    write(temp_file, supercell)
                    
                    # Update the file to process
                    f = temp_file
                    stem = f"{stem}_{nx}x{ny}x{nz}"
                    outp = out/(stem+ext)
                    
                    print(f"Supercell created with {len(supercell)} atoms")
                except Exception as e:
                    print(f"Error creating supercell: {e}")
                    print(f"Proceeding with original structure for {f.name}")
            
            # VASP-specific processing
            if fmt == "vasp":
                # Create a directory for this structure
                struct_dir = out / stem
                struct_dir.mkdir(parents=True, exist_ok=True)
                
                try:
                    # For CIF files, convert to XYZ first
                    if f.suffix.lower() == '.cif':
                        xyz_path = out / f"{stem}.xyz"
                        if cif_to_xyz(f, xyz_path):
                            self._copy_to_xyz_checks(xyz_path, out)
                            # This is the critical fix - include kpoints!
                            result = self._generate_vasp_input_for_file(xyz_path, struct_dir, params, kpoints)
                            # Delete the temporary XYZ file after use
                            if xyz_path.exists():
                                xyz_path.unlink()
                            print("→", "OK" if result else "✗", f.name)
                    else:
                        # Direct XYZ processing
                        result = self._generate_vasp_input_for_file(f, struct_dir, params, kpoints)
                        print("→", "OK" if result else "✗", f.name)
                except Exception as e:
                    print(f"→ ✗ Error processing {f.name}: {e}")
                
        # Clean up temporary directory if it was created
        if use_supercell and (out / "temp_supercell").exists():
            import shutil
            shutil.rmtree(out / "temp_supercell")

    def _get_vasp_custom_params(self):
        """Get custom parameters for VASP."""
        params = {}
        params["SYSTEM"] = input("SYSTEM description: ") or "Custom calculation"
        try:
            params["ENCUT"] = float(input("ENCUT [400]: ") or 400)
            params["ISMEAR"]= int(input("ISMEAR [0]: ") or 0)
            params["SIGMA"] = float(input("SIGMA [0.05]: ") or 0.05)
        except:
            params["ENCUT"],params["ISMEAR"],params["SIGMA"] = 400,0,0.05
        relax = input("Ionic relaxation? (y/n) [y]: ").strip().lower() or "y"
        if relax=="y":
            try:
                params["IBRION"] = int(input("IBRION [2]: ") or 2)
                params["NSW"]    = int(input("NSW [100]: ") or 100)
                params["ISIF"]   = int(input("ISIF [3]: ") or 3)
                params["EDIFFG"] = float(input("EDIFFG [-0.01]: ") or -0.01)
            except:
                params.update({"IBRION":2,"NSW":100,"ISIF":3,"EDIFFG":-0.01})
        else:
            params.update({"IBRION":-1,"NSW":0,"ISIF":2})
        params["PREC"]  = input("PREC [Accurate]: ") or "Accurate"
        try:
            params["EDIFF"]= float(input("EDIFF [1e-5]: ") or 1e-5)
        except:
            params["EDIFF"]=1e-5
        params["LWAVE"]= input("LWAVE (y/n) [n]: ").lower()=="y"
        params["LCHARG"]= input("LCHARG (y/n) [y]: ").lower()!="n"
        ktype = input("KPOINTS type (gamma/monkhorst) [gamma]: ") or "gamma"
        try:
            mesh = [int(x) for x in (input("Grid [3 3 3]: ") or "3 3 3").split()]
        except:
            mesh=[3,3,3]
        params["KPOINTS"]={"type":ktype,"grid":mesh}
        return params

    def _process_vasp_structure_file(self, file_path, vasp_params, out_dir):
        """Process a single structure file for VASP input generation."""
        ok,msg,_ = validate_structure(file_path)
        if not ok:
            print(f"Skipping {file_path.name}: {msg}")
            return False
        
        try:
            # Generate INCAR
            incar_content = self._generate_vasp_incar(vasp_params)
            
            # Parse structure data
            if file_path.name.lower().endswith('.cif'):
                from multi_agent_dft.file_processing.cif import parse_cif_file
                structure_data = parse_cif_file(file_path)
            else:  # XYZ file
                from multi_agent_dft.file_processing.xyz import parse_xyz
                structure_data = parse_xyz(file_path)
            
            if not structure_data:
                print(f"Error: Could not parse structure file {file_path}")
                return False
            
            # Generate POSCAR
            poscar_content = self._generate_vasp_poscar(structure_data)
            
            # Generate KPOINTS
            kpoints_content = self._generate_vasp_kpoints(vasp_params)
            
            # Save files
            base_name = file_path.stem
            with open(out_dir / f"{base_name}_INCAR", "w") as f:
                f.write(incar_content)
            with open(out_dir / f"{base_name}_POSCAR", "w") as f:
                f.write(poscar_content)
            with open(out_dir / f"{base_name}_KPOINTS", "w") as f:
                f.write(kpoints_content)
            
            return True
            
        except Exception as e:
            print(f"Error generating VASP input files for {file_path.name}: {str(e)}")
            return False
    
    def _generate_vasp_incar(self, params):
        """Generate VASP INCAR file content from parameters."""
        incar_lines = []
        
        # Skip these special parameters
        skip_params = ["template_name", "KPOINTS"]
        
        # Add system description first
        if "SYSTEM" in params:
            incar_lines.append(f"SYSTEM = {params['SYSTEM']}")
        
        # Add all other parameters
        for key, value in sorted(params.items()):
            if key in skip_params or key == "SYSTEM":
                continue
                
            # Format value based on type
            if isinstance(value, bool):
                incar_lines.append(f"{key} = {'.TRUE.' if value else '.FALSE.'}")
            elif isinstance(value, dict):
                # Skip dictionaries
                continue
            else:
                incar_lines.append(f"{key} = {value}")
        
        return "\n".join(incar_lines)
    
    def _generate_vasp_poscar(self, structure_data):
        """Generate VASP POSCAR file content from structure data."""
        poscar_lines = []
        
        # Title line
        poscar_lines.append(f"Generated by Multi-Agent DFT System - {structure_data['meta']['filename']}")
        
        # Scaling factor
        poscar_lines.append("1.0")
        
        # Lattice vectors
        cell = structure_data.get('cell', [
            [10.0, 0.0, 0.0], 
            [0.0, 10.0, 0.0], 
            [0.0, 0.0, 10.0]
        ])
        
        for vector in cell:
            poscar_lines.append(f"  {vector[0]:15.10f}  {vector[1]:15.10f}  {vector[2]:15.10f}")
        
        # Get unique elements and their counts
        elements = {}
        for atom in structure_data['atoms']:
            symbol = atom['symbol']
            if symbol not in elements:
                elements[symbol] = 0
            elements[symbol] += 1
        
        # Element symbols line
        poscar_lines.append("  " + "  ".join(elements.keys()))
        
        # Element counts line
        poscar_lines.append("  " + "  ".join(str(count) for count in elements.values()))
        
        # Direct or Cartesian
        poscar_lines.append("Cartesian")
        
        # Sort atoms by element
        for element in elements.keys():
            for atom in structure_data['atoms']:
                if atom['symbol'] == element:
                    pos = atom['position']
                    poscar_lines.append(f"  {pos[0]:15.10f}  {pos[1]:15.10f}  {pos[2]:15.10f}")
        
        return "\n".join(poscar_lines)

    def _generate_vasp_kpoints(self, kpoints_config):
        """Generate VASP KPOINTS file content from parameters."""
        kpoints_lines = []
        
        # Get kpoints configuration
        if isinstance(kpoints_config, dict):
            # It's already a dictionary configuration
            pass
        else:
            # It might be in the vasp_config
            kpoints_config = getattr(kpoints_config, 'KPOINTS', {'type': 'gamma', 'grid': [3, 3, 3]})
        
        # Comment line
        kpoints_lines.append("Automatic k-point generation")
        
        # Handle explicit k-points
        if kpoints_config.get('type') == 'explicit' and 'explicit_points' in kpoints_config:
            kpoints_lines[0] = "Explicit k-points"
            # Set number of k-points
            kpoints_lines.append(str(kpoints_config.get('num_kpoints', 0)))
            # Set coordinate system (cartesian or reciprocal)
            kpoints_lines.append("Cartesian")  # or "Reciprocal" if needed
            
            # Add each k-point
            for point in kpoints_config['explicit_points']:
                kx, ky, kz, weight = point
                kpoints_lines.append(f"{kx:12.8f} {ky:12.8f} {kz:12.8f} {weight:12.8f}")
                
            return "\n".join(kpoints_lines)
        
        # Number of k-points (0 for automatic generation)
        kpoints_lines.append("0")
        
        # Type of k-points
        k_type = kpoints_config.get('type', 'gamma').lower()
        
        if k_type == 'gamma':
            kpoints_lines.append("Gamma")
        elif k_type == 'monkhorst':
            kpoints_lines.append("Monkhorst-Pack")
        elif k_type == 'line':
            kpoints_lines.append("Line-mode")
            # Special handling for band structure calculations would go here
            return "\n".join(kpoints_lines + ["Reciprocal", "20  ! 20 points per line segment", "Line-mode", "Reciprocal"])
        else:
            # Default to gamma
            kpoints_lines.append("Gamma")
        
        # K-points grid
        grid = kpoints_config.get('grid', [3, 3, 3])
        kpoints_lines.append(f"{grid[0]} {grid[1]} {grid[2]}")
        
        # Shift (usually 0 0 0)
        kpoints_lines.append("0 0 0")
        
        return "\n".join(kpoints_lines)


##### GAUSSIAN INPUT GENERATION METHODS ##########

    def _handle_gaussian_input_generation(self, use_supercell=False, supercell_dims=None):
        """Handle Gaussian input generation with detailed step-by-step guidance."""
        print("\n==== Gaussian Input File Generation (Guided Mode) ====\n")
        
        # Step 1: Get structure file path
        file_path = input("Enter the path to your structure file or directory (XYZ or CIF): ")
        input_path = Path(file_path).expanduser().resolve()
        
        # Validate the input file/directory
        if not input_path.exists():
            print(f"Error: Path '{input_path}' does not exist.")
            return
        
        # Step 2: Output location
        output_path = input("Enter the output directory path where input files will be generated: ")
        output_dir = Path(output_path).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Step 3: If directory - ask for batch processing or select specific file
        if input_path.is_dir():
            cif_files = list(input_path.glob("*.cif"))
            xyz_files = list(input_path.glob("*.xyz"))
            
            if not (cif_files or xyz_files):
                print(f"No .cif or .xyz files found in {input_path}")
                return
            
            print(f"\nFound {len(cif_files)} CIF files and {len(xyz_files)} XYZ files.")
            process_mode = input("Process all files or select a specific file? (all/select): ").strip().lower()
            
            if process_mode == "select":
                # Show files for selection
                all_files = cif_files + xyz_files
                for i, f in enumerate(all_files, 1):
                    print(f"{i}. {f.name}")
                
                try:
                    selected = int(input("\nSelect file number: "))
                    if 1 <= selected <= len(all_files):
                        input_path = all_files[selected-1]
                    else:
                        print("Invalid selection. Using first file.")
                        input_path = all_files[0]
                except ValueError:
                    print("Invalid input. Using first file.")
                    input_path = all_files[0]
                    
                print(f"\nSelected file: {input_path.name}")
                
                # Now input_path is a single file
                # Process with supercell if requested
                if use_supercell and supercell_dims:
                    self._process_supercell_gaussian_file(input_path, output_dir, supercell_dims)
                else:
                    self._process_single_gaussian_file(input_path, output_dir)
            else:
                # Proceed with batch processing
                print("\nBatch processing all structure files...\n")
                
                # Step 4: Get CP2K parameters (once for all files)
                gaussian_config = self._get_gaussian_parameters_interactively()
                
                # Process all files
                processed = 0
                
                if cif_files:
                    print(f"\nProcessing {len(cif_files)} CIF files...")
                    for cif_file in cif_files:
                        try:
                            print(f"  Processing {cif_file.name}...")
                            
                            # Handle supercell creation if requested
                            if use_supercell and supercell_dims:
                                self._process_supercell_gaussian_file(cif_file, output_dir, supercell_dims, gaussian_config)
                            else:
                                # Original processing
                                xyz_path = output_dir / f"{cif_file.stem}.xyz"
                                if cif_to_xyz(cif_file, xyz_path):
                                    self._copy_to_xyz_checks(xyz_path, output_dir)
                                    processed += 1
                                    self._generate_gaussian_input_for_file(xyz_path, output_dir, gaussian_config)
                            
                            processed += 1
                        except Exception as e:
                            print(f"  Error processing {cif_file.name}: {e}")
                
                if xyz_files:
                    print(f"\nProcessing {len(xyz_files)} XYZ files...")
                    for xyz_file in xyz_files:
                        try:
                            print(f"  Processing {xyz_file.name}...")
                            
                            # Handle supercell creation if requested
                            if use_supercell and supercell_dims:
                                self._process_supercell_gaussian_file(xyz_file, output_dir, supercell_dims, gaussian_config)
                            else:
                                # Original processing
                                self._copy_to_xyz_checks(xyz_file, output_dir)
                                processed += 1
                                self._generate_gaussian_input_for_file(xyz_file, output_dir, gaussian_config)
                        except Exception as e:
                            print(f"  Error processing {xyz_file.name}: {e}")
                
                print(f"\nProcessed {processed} files. Gaussian input files generated in {output_dir}")
        else:
            # Process single file
            if use_supercell and supercell_dims:
                self._process_supercell_gaussian_file(input_path, output_dir, supercell_dims)
            else:
                self._process_single_gaussian_file(input_path, output_dir)

        self._cleanup_xyz_files(output_dir)
        self._verify_output_files(output_dir)

    def _process_single_gaussian_file(self, file_path, output_dir):
        """Process a single file for Gaussian input generation."""
        # Validate the file
        valid, msg, _ = validate_structure(file_path)
        if not valid:
            print(f"Structure validation failed: {msg}")
            return
        
        # For CIF files, convert to XYZ first
        if file_path.suffix.lower() == ".cif":
            print(f"\nConverting CIF file to XYZ format...")
            xyz_path = output_dir / f"{file_path.stem}.xyz"
            if not cif_to_xyz(file_path, xyz_path):
                print("CIF conversion failed. Aborting.")
                return
            file_path = xyz_path
            print(f"Conversion successful: {xyz_path}")
            self._copy_to_xyz_checks(xyz_path, output_dir)
        
        # Get Gaussian parameters through interactive prompts
        gaussian_config = self._get_gaussian_parameters_interactively()
        
        # Generate Gaussian input file
        self._generate_gaussian_input_for_file(file_path, output_dir, gaussian_config)
        
        print(f"\nGaussian input generated in {output_dir}")

    def _process_supercell_gaussian_file(self, file_path, output_dir, supercell_dims, gaussian_config=None):
        """Process a single file with supercell creation for Gaussian input generation."""
        nx, ny, nz = supercell_dims
        
        # Validate the file
        valid, msg, _ = validate_structure(file_path)
        if not valid:
            print(f"Structure validation failed: {msg}")
            return
        
        # Create a temporary directory for supercell processing
        temp_dir = output_dir / "temp_supercell"
        temp_dir.mkdir(exist_ok=True)
        
        try:
            from ase.io import read, write
            from ase.build import make_supercell
            import numpy as np
            
            # For CIF files, convert to XYZ first if needed
            if file_path.suffix.lower() == ".cif":
                print(f"\nConverting CIF file to XYZ format...")
                xyz_path = temp_dir / f"{file_path.stem}.xyz"
                if not cif_to_xyz(file_path, xyz_path):
                    print("CIF conversion failed. Aborting.")
                    return
                file_path = xyz_path
                print(f"Conversion successful: {xyz_path}")
                self._copy_to_xyz_checks(xyz_path, output_dir)
            
            # Create the supercell
            print(f"Creating {nx}x{ny}x{nz} supercell...")
            ase_struct = read(file_path)
            NxNxN = np.array([nx, ny, nz])
            supercell = make_supercell(ase_struct, np.eye(3) * NxNxN)
            
            # Save the supercell to a temporary file
            supercell_file = temp_dir / f"{file_path.stem}_{nx}x{ny}x{nz}.xyz"
            write(supercell_file, supercell)
            print(f"Supercell created with {len(supercell)} atoms")
            
            # Get Gaussian parameters if not provided
            if gaussian_config is None:
                gaussian_config = self._get_gaussian_parameters_interactively()
            
            # Generate Gaussian input for the supercell
            output_file = output_dir / f"{file_path.stem}_{nx}x{ny}x{nz}.inp"
            
            # Parse XYZ file and generate CP2K input
            from multi_agent_dft.file_processing.xyz import parse_xyz
            from multi_agent_dft.dft.gaussian import save_gaussian_input
            
            structure_data = parse_xyz(supercell_file)
            if structure_data:
                success = save_gaussian_input(structure_data, output_file, gaussian_config)
                if success:
                    print(f"Generated Gaussian input file: {output_file}")
                else:
                    print(f"Failed to generate Gaussian input file for {supercell_file}")
            else:
                print(f"Error: Could not parse XYZ file {supercell_file}")
            
        except Exception as e:
            print(f"Error in supercell creation or Gaussian input generation: {e}")
        finally:
            # Clean up temporary directory
            import shutil
            shutil.rmtree(temp_dir)

    def _generate_gaussian_input_for_file(self, xyz_file, output_dir, gaussian_config):
        """Generate a Gaussian input file for a single XYZ file."""
        from multi_agent_dft.file_processing.xyz import parse_xyz
        from multi_agent_dft.dft.gaussian import save_gaussian_input
        
        try:
            # Parse XYZ file
            structure_data = parse_xyz(xyz_file)
            if not structure_data:
                print(f"Error: Could not parse XYZ file {xyz_file}")
                return False
            
            # Generate Gaussian input file with special handling for force printing
            # We need to modify the input generation to include force printing
            output_file = output_dir / f"{xyz_file.stem}.gjf"
            
            # Add force and printing-related options to the route section
            custom_config = gaussian_config.copy()
            route_additions = []
            
            # Handle printing forces
            if gaussian_config['print_options'].get('forces', False):
                route_additions.append("Force")
                
            # Handle force constants
            if gaussian_config['print_options'].get('force_constants', False):
                route_additions.append("ForceCon")
                
            # Handle output verbosity
            if gaussian_config['print_options'].get('debug', False):
                route_additions.append("Output=Debug")
            elif gaussian_config['print_options'].get('verbose', False):
                route_additions.append("Output=Verbose")
                
            # Handle checkpoint file
            if gaussian_config['print_options'].get('save_chk', False):
                # Also add the %chk line at the beginning of the file
                custom_config['save_chk'] = f"{xyz_file.stem}.chk"
            
            # Store the route additions
            if route_additions:
                custom_config['route_additions'] = route_additions
            
            # Call the save_gaussian_input function with our modified config
            success = save_gaussian_input(structure_data, output_file, custom_config)
            
            if success:
                print(f"  Generated Gaussian input file: {output_file}")
                return True
            else:
                print(f"  Failed to generate Gaussian input file for {xyz_file}")
                return False
        except Exception as e:
            print(f"  Error generating Gaussian input for {xyz_file}: {e}")
            return False

    def _get_gaussian_parameters_interactively(self):
        """Interactive questionnaire for Gaussian input parameters."""
        print("\n=== Gaussian Calculation Settings ===")
        
        # Initialize config
        gaussian_config = {
            'method': 'B3LYP',
            'basis_set': '6-31G(d)',
            'job_type': 'Opt',
            'charge': 0,
            'multiplicity': 1,
            'memory': '4GB',
            'nproc': 4,
            'print_options': {}
        }
        
        # 1. Basic settings
        print("\n--- Basic Settings ---")
        memory = input("Memory allocation (e.g., 4GB, 8GB) [4GB]: ").strip() or "4GB"
        gaussian_config['memory'] = memory
        
        num_procs = input("Number of processors to use [4]: ").strip() or "4"
        try:
            gaussian_config['nproc'] = int(num_procs)
        except ValueError:
            print("Invalid value. Using default 4.")
            gaussian_config['nproc'] = 4
        
        # 2. Method and basis set
        print("\n--- Computational Method ---")
        
        print("\nAvailable methods:")
        methods = ["B3LYP", "ωB97X-D", "M06-2X", "PBE0", "CAM-B3LYP", "MP2", "HF", "PBE"]
        for i, method in enumerate(methods, 1):
            print(f"{i}. {method}")
        
        method_choice = input("Select method (1-8) [1]: ").strip() or "1"
        try:
            idx = int(method_choice) - 1
            gaussian_config['method'] = methods[idx]
        except (ValueError, IndexError):
            print("Invalid selection. Using B3LYP.")
            gaussian_config['method'] = "B3LYP"
        
        print("\nAvailable basis sets:")
        basis_sets = ["6-31G(d)", "6-31+G(d,p)", "6-311G(d,p)", "6-311++G(2d,2p)", "cc-pVDZ", "cc-pVTZ", "def2-SVP", "def2-TZVP"]
        for i, basis in enumerate(basis_sets, 1):
            print(f"{i}. {basis}")
        
        basis_choice = input("Select basis set (1-8) [1]: ").strip() or "1"
        try:
            idx = int(basis_choice) - 1
            gaussian_config['basis_set'] = basis_sets[idx]
        except (ValueError, IndexError):
            print("Invalid selection. Using 6-31G(d).")
            gaussian_config['basis_set'] = "6-31G(d)"
        
        # 3. Job type
        print("\n--- Calculation Type ---")
        print("\nAvailable job types:")
        job_types = ["Opt", "Freq", "OptFreq", "SP", "SP NMR", "TD", "Opt Freq"]
        for i, job in enumerate(job_types, 1):
            print(f"{i}. {job} - {self._get_job_description(job)}")
        
        job_choice = input("Select job type (1-7) [1]: ").strip() or "1"
        try:
            idx = int(job_choice) - 1
            gaussian_config['job_type'] = job_types[idx]
        except (ValueError, IndexError):
            print("Invalid selection. Using Opt.")
            gaussian_config['job_type'] = "Opt"
        
        # 4. Molecular properties
        print("\n--- Molecular Properties ---")
        
        try:
            charge = input("Molecular charge [0]: ").strip() or "0"
            gaussian_config['charge'] = int(charge)
        except ValueError:
            print("Invalid value. Using default 0.")
            gaussian_config['charge'] = 0
        
        try:
            multiplicity = input("Spin multiplicity (1=singlet, 2=doublet, etc.) [1]: ").strip() or "1"
            gaussian_config['multiplicity'] = int(multiplicity)
        except ValueError:
            print("Invalid value. Using default 1.")
            gaussian_config['multiplicity'] = 1
        
        # 5. Solvent effects
        print("\n--- Solvent Effects ---")
        use_solvent = input("Include solvent model? (y/n) [n]: ").strip().lower() or "n"
        
        if use_solvent == "y":
            print("\nCommon solvents:")
            solvents = ["Water", "Chloroform", "Acetonitrile", "Methanol", "Benzene", "Toluene", "Dichloromethane", "DMSO", "Hexane", "Ethanol"]
            for i, solvent in enumerate(solvents, 1):
                print(f"{i}. {solvent}")
            
            solvent_choice = input("Select solvent (1-10) or type name: ").strip()
            try:
                idx = int(solvent_choice) - 1
                if 0 <= idx < len(solvents):
                    gaussian_config['solvent'] = solvents[idx]
                else:
                    print("Invalid choice. Using Water.")
                    gaussian_config['solvent'] = "Water"
            except ValueError:
                # User entered a custom solvent name
                if solvent_choice:
                    gaussian_config['solvent'] = solvent_choice
                else:
                    gaussian_config['solvent'] = "Water"
            
            print("\nSolvent models:")
            models = ["PCM", "SMD", "CPCM"]
            for i, model in enumerate(models, 1):
                print(f"{i}. {model}")
            
            model_choice = input("Select solvent model (1-3) [1]: ").strip() or "1"
            try:
                idx = int(model_choice) - 1
                if 0 <= idx < len(models):
                    gaussian_config['solvent_model'] = models[idx]
                else:
                    gaussian_config['solvent_model'] = "PCM"
            except ValueError:
                gaussian_config['solvent_model'] = "PCM"
        
        # 6. Output and printing options
        print("\n--- Output and Printing Options ---")
        
        # Forces printing - matching the CP2K implementation
        print_forces = input("Print forces for each calculation step? (y/n) [y]: ").strip().lower() or "y"
        if print_forces == "y":
            # For Gaussian, we add keywords to print forces
            gaussian_config['print_options']['forces'] = True
        
        # Output file options
        print("\nOutput file options:")
        print("1. Normal output")
        print("2. Verbose output (detailed SCF and optimization steps)")
        print("3. Debug output (maximum detail, large files)")
        
        output_detail = input("Select output level (1-3) [1]: ").strip() or "1"
        try:
            output_level = int(output_detail)
            if output_level == 2:
                gaussian_config['print_options']['verbose'] = True
            elif output_level == 3:
                gaussian_config['print_options']['debug'] = True
        except ValueError:
            print("Invalid selection. Using normal output.")
        
        # Save checkpoint file
        save_chk = input("Save checkpoint file for restart capability? (y/n) [y]: ").strip().lower() or "y"
        if save_chk == "y":
            gaussian_config['print_options']['save_chk'] = True
        
        # 7. Advanced options
        print("\n--- Advanced Options ---")
        use_advanced = input("Configure advanced options? (y/n) [n]: ").strip().lower() or "n"
        
        if use_advanced == "y":
            # SCF Convergence
            scf_options = input("Tighter SCF convergence? (y/n) [n]: ").strip().lower() or "n"
            if scf_options == "y":
                gaussian_config['scf'] = "tight"
            
            # Population analysis
            pop_analysis = input("Include population analysis? (None/NBO/MK/Hirshfeld) [None]: ").strip() or "None"
            if pop_analysis.lower() != "none":
                gaussian_config['pop'] = pop_analysis
            
            # Dispersion correction (if not already included in method)
            if gaussian_config['method'] not in ["ωB97X-D", "M06-2X"]:
                add_dispersion = input("Add empirical dispersion correction? (y/n) [n]: ").strip().lower() or "n"
                if add_dispersion == "y":
                    gaussian_config['dispersion'] = "GD3"
                
            # Force constants
            if any(x in gaussian_config['job_type'] for x in ['Opt', 'Freq']):
                print_force_constants = input("Print force constants? (y/n) [n]: ").strip().lower() or "n"
                if print_force_constants == "y":
                    gaussian_config['print_options']['force_constants'] = True
        
        # Summary of selected options
        print("\n=== Summary of Gaussian Calculation Settings ===")
        print(f"Method:          {gaussian_config['method']}")
        print(f"Basis Set:       {gaussian_config['basis_set']}")
        print(f"Job Type:        {gaussian_config['job_type']}")
        print(f"Charge:          {gaussian_config['charge']}")
        print(f"Multiplicity:    {gaussian_config['multiplicity']}")
        print(f"Memory:          {gaussian_config['memory']}")
        print(f"Processors:      {gaussian_config['nproc']}")
        
        if 'solvent' in gaussian_config:
            print(f"Solvent:         {gaussian_config['solvent']} ({gaussian_config.get('solvent_model', 'PCM')})")
        
        if 'scf' in gaussian_config:
            print(f"SCF Convergence: Tight")
        
        if 'pop' in gaussian_config:
            print(f"Population:      {gaussian_config['pop']}")
        
        if 'dispersion' in gaussian_config:
            print(f"Dispersion:      {gaussian_config['dispersion']}")
        
        # Print output options
        print("\nOutput Options:")
        if gaussian_config['print_options'].get('forces', False):
            print("- Print forces for each step")
        if gaussian_config['print_options'].get('force_constants', False):
            print("- Print force constants")
        if gaussian_config['print_options'].get('verbose', False):
            print("- Verbose output")
        if gaussian_config['print_options'].get('debug', False):
            print("- Debug level output")
        if gaussian_config['print_options'].get('save_chk', False):
            print("- Save checkpoint file")
        
        confirm = input("\nConfirm these settings? (y/n) [y]: ").strip().lower() or "y"
        if confirm != "y":
            print("Restarting parameter selection...")
            return self._get_gaussian_parameters_interactively()
        
        return gaussian_config

    def _get_job_description(self, job_type):
        """Return a description for each job type."""
        descriptions = {
            "Opt": "Geometry optimization",
            "Freq": "Vibrational frequency calculation",
            "OptFreq": "Geometry optimization followed by frequency calculation",
            "SP": "Single point energy calculation",
            "SP NMR": "NMR chemical shift calculation",
            "TD": "Time-dependent calculation for excited states",
            "Opt Freq": "Geometry optimization followed by frequency calculation",
        }
        return descriptions.get(job_type, "")

    def _process_gaussian_file(self, file_path, gaussian_config, out_dir):
        """Process a single file for Gaussian input generation."""
        try:
            # Import required modules
            from multi_agent_dft.file_processing.xyz import parse_xyz
            
            # Parse XYZ file
            structure_data = None
            if file_path.suffix.lower() == '.xyz':
                structure_data = parse_xyz(file_path)
            elif file_path.suffix.lower() == '.cif':
                from multi_agent_dft.file_processing.cif import parse_cif_file
                structure_data = parse_cif_file(file_path)
            
            if not structure_data:
                print(f"Failed to parse structure file: {file_path}")
                return False
            
            # Generate Gaussian input
            gjf_content = self._generate_gaussian_input(structure_data, gaussian_config)
            
            # Save the file
            output_file = out_dir / f"{file_path.stem}.com"
            with open(output_file, 'w') as f:
                f.write(gjf_content)
            
            print(f"Generated Gaussian input file: {output_file}")
            return True
        
        except Exception as e:
            print(f"Error processing file {file_path}: {str(e)}")
            return False
    
    def _generate_gaussian_input(self, structure_data, config):
        """Generate Gaussian input file content."""
        gjf_lines = []
        
        # Memory and processor specification
        mem = config.get('memory', '4GB')
        nproc = config.get('nproc', '4')
        gjf_lines.append(f"%mem={mem}")
        gjf_lines.append(f"%nproc={nproc}")
        
        # Route section
        method = config.get('method', 'B3LYP')
        basis = config.get('basis_set', '6-31G(d)')
        job = config.get('job_type', 'Opt')
        
        route = f"#p {method}/{basis} {job}"
        
        # Add solvent if specified
        if 'solvent' in config:
            route += f" SCRF=(Solvent={config['solvent']})"
        
        gjf_lines.append(route)
        gjf_lines.append("")  # Empty line
        
        # Title section
        gjf_lines.append(f"Generated by Multi-Agent DFT System - {structure_data['meta']['filename']}")
        gjf_lines.append("")  # Empty line
        
        # Charge and multiplicity
        charge = config.get('charge', 0)
        multiplicity = config.get('multiplicity', 1)
        gjf_lines.append(f"{charge} {multiplicity}")
        
        # Atomic coordinates
        for atom in structure_data['atoms']:
            pos = atom['position']
            gjf_lines.append(f"{atom['symbol']} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}")
        
        gjf_lines.append("")  # Empty line
        
        return "\n".join(gjf_lines)

    def process_query(self, query):
        """Process a specific research query non-interactively."""
        print(f"\nProcessing research query: {query}")
        
        # Generate a default follow-up context
        followup_question = self.sup1_agent.generate_followup_question(query)
        print(f"\nFollow-up question: {followup_question}")
        additional_context = ""  # No user input in non-interactive mode
        
        # Refine the query
        refined_query = self.refine_query(query, additional_context)
        print(f"\nRefined research query: {refined_query}")
        
        # Get summaries from the chemistry agents
        print("\nAnalyzing research literature...")
        exp_summary = self.exp_agent.summarize(refined_query, additional_context)
        theo_summary = self.theo_agent.summarize(refined_query, additional_context)
        
        print("\n--- Experimental Chemist Summary ---")
        print(exp_summary)
        print("\n--- Theoretical Chemist Summary ---")
        print(theo_summary)
        
        # Integrate the chemistry summaries
        integrated_content = f"Experimental Summary:\n{exp_summary}\n\nTheoretical Summary:\n{theo_summary}"
        sup1_report = self.sup1_agent.integrate(integrated_content)
        print("\n--- Integrated Scientific Report ---")
        print(sup1_report)
        
        # DFT analysis
        print("\nEngaging DFT Expert Agents...")
        gaussian_report = self.gaussian_expert.analyze(refined_query)
        vasp_report = self.vasp_expert.analyze(refined_query)
        cp2k_report = self.cp2k_expert.analyze(refined_query)
        
        # DFT recommendation
        dft_content = f"GAUSSIAN Report:\n{gaussian_report}\n\nVASP Report:\n{vasp_report}\n\nCP2K Report:\n{cp2k_report}"
        dft_recommendation = self.sup2_agent.integrate(dft_content)
        print("\n--- DFT Recommendation ---")
        print(dft_recommendation)
        
        # Save reports to files
        output_dir = Path(f"research_output_{query.replace(' ', '_')[:30]}").resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_dir / "experimental_summary.md", "w") as f:
            f.write(f"# Experimental Summary for: {query}\n\n{exp_summary}")
        
        with open(output_dir / "theoretical_summary.md", "w") as f:
            f.write(f"# Theoretical Summary for: {query}\n\n{theo_summary}")
        
        with open(output_dir / "integrated_report.md", "w") as f:
            f.write(f"# Integrated Report for: {query}\n\n{sup1_report}")
        
        with open(output_dir / "dft_recommendation.md", "w") as f:
            f.write(f"# DFT Recommendation for: {query}\n\n{dft_recommendation}")
        
        print(f"\nReports saved to: {output_dir}")
        print("\nProcess completed. Thank you for using the Multi-Agent DFT Research System.")
   
    def _copy_to_xyz_checks(self, file_path, output_dir, cleanup=True):  # Set default to False!
        """
        Copy an XYZ file to the xyz_checks directory for later inspection.
        Does NOT delete original files by default.
        """
        from shutil import copy2
        
        # Convert to Path objects
        file_path = Path(file_path)
        output_dir = Path(output_dir)
        
        # Only process .xyz files
        if file_path.suffix.lower() != ".xyz":
            print(f"  Skipping non-XYZ file: {file_path.name}")
            return

        # Create xyz_checks directory
        xyz_checks_dir = output_dir / "xyz_checks"
        xyz_checks_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Copy file to xyz_checks
            copy2(file_path, xyz_checks_dir / file_path.name)
            print(f"  Copied {file_path.name} → xyz_checks/")
            
            # We won't delete anything here anymore
        except Exception as e:
            print(f"  Error copying file {file_path.name}: {e}")
   
####### OUTPUT FILE VERIFICATION AND CLEANUP METHODS ######

    def _verify_output_files(self, output_dir):
        """Verify that all input files are still present in the output directory."""
        output_dir = Path(output_dir)
        
        # list all input files in the directory - check for both .inp files and VASP directories
        inp_files = list(output_dir.glob("*.inp"))
        vasp_dirs = [d for d in output_dir.iterdir() if d.is_dir() and not d.name == "xyz_checks"]
        
        if not inp_files and not vasp_dirs:
            print("WARNING: No input files found in output directory! Something may have gone wrong.")
            print(f"Output directory contents: {[f.name for f in output_dir.iterdir()]}")
        else:
            total_files = len(inp_files) + len(vasp_dirs)
            print(f"Verification successful: {total_files} input files/directories found in output directory.")

    def _cleanup_xyz_files(self, output_dir):
        """
        Remove all XYZ files from the output directory AFTER all input generation is complete.
        This should be called only at the very end of processing.
        """
        output_dir = Path(output_dir)
        
        print("\nCleaning up XYZ files from output directory...")
        try:
            # Find all XYZ files in the root of the output directory
            xyz_files = list(output_dir.glob("*.xyz"))
            
            if not xyz_files:
                print("  No XYZ files to clean up.")
                return
                
            for xyz_file in xyz_files:
                try:
                    # Make sure this is not within xyz_checks directory
                    if "xyz_checks" not in str(xyz_file):
                        xyz_file.unlink()
                        print(f"  Removed: {xyz_file.name}")
                except Exception as e:
                    print(f"  Failed to remove {xyz_file.name}: {e}")
                    
            print(f"Cleanup complete. Removed {len(xyz_files)} XYZ files.")
        except Exception as e:
            print(f"  Error during cleanup: {e}")

            # Call this after cleanup
        self._verify_clean_directory(output_dir)

    def _verify_clean_directory(self, output_dir):
        """Verify the output directory is clean of temporary XYZ files."""
        output_dir = Path(output_dir)
        
        # Double-check for any remaining XYZ files outside xyz_checks
        remaining_xyz = list(output_dir.glob("*.xyz"))
        
        if remaining_xyz:
            print("WARNING: Found remaining XYZ files that should have been cleaned up:")
            for xyz in remaining_xyz:
                print(f"  - {xyz.name}")
                # Option to force cleanup
                try:
                    xyz.unlink()
                    print(f"    Removed: {xyz.name}")
                except Exception as e:
                    print(f"    Failed to remove: {e}")
        else:
            print("Directory is clean - all temporary XYZ files have been properly removed.")

####### DFT OUTPUT PROCESSING METHODS ######

    def _handle_output_processing(self):
        """Handle processing of DFT output files to extract forces, energies, and coordinates."""
        print("\n==== DFT Output Processing ====\n")
        
        # Step 1: Ask for the DFT code type
        print("Available DFT codes:")
        print("1. CP2K")
        print("2. VASP (Enhanced - processes all optimization steps)")
        print("3. Gaussian")
        
        code_choice = input("\nSelect DFT code (1/2/3): ").strip()
        
        if code_choice == "1":
            code_type = "cp2k"
        elif code_choice == "2":
            # Use enhanced VASP processing
            self._handle_vasp_output_processing_enhanced()
            return
        elif code_choice == "3":
            code_type = "gaussian"
        else:
            print("Invalid choice. Defaulting to CP2K.")
            code_type = "cp2k"
        
        # Original processing for CP2K and Gaussian
        if code_type == "cp2k":
            print("\n--- CP2K Output Processing ---")
            input_file = input("Path to CP2K input file (.inp): ").strip()
            output_file = input("Path to CP2K output file: ").strip()
            frac_xyz_file = input("Path to fractional coordinates XYZ file: ").strip()
            output_json = input("Path for output JSON file [output_data.json]: ").strip() or "output_data.json"
            
            # Option for coordinate conversion
            do_frac_to_cart = input("Convert fractional to Cartesian coordinates? (y/n) [y]: ").strip().lower() != "n"
            
            try:
                process_dft_output(
                    code_type,
                    input_file=input_file,
                    output_file=output_file,
                    frac_xyz_file=frac_xyz_file,
                    output_json=output_json,
                    do_frac_to_cart=do_frac_to_cart
                )
                print("\nCP2K output processing completed successfully.")
            except Exception as e:
                print(f"\nError processing CP2K output: {e}")
                
        elif code_type == "gaussian":
            print("\n--- Gaussian Output Processing ---")
            log_file = input("Path to Gaussian log file: ").strip()
            output_json = input("Path for output JSON file [output_data.json]: ").strip() or "output_data.json"
            
            try:
                process_dft_output(
                    code_type,
                    log_file=log_file,
                    output_json=output_json
                )
                print("\nGaussian output processing completed successfully.")
            except Exception as e:
                print(f"\nError processing Gaussian output: {e}")
        
        print("\nOutput processing completed.")
        
        # Ask if the user wants to process another output
        another = input("\nProcess another output? (y/n): ").strip().lower()
        if another == "y":
            self._handle_output_processing()

    def _handle_vasp_output_processing_enhanced(self):
        """Enhanced VASP output processing with batch capabilities."""
        print("\n==== Enhanced VASP Output Processing ====\n")
        
        # Import the enhanced processor
        from pathlib import Path
        import json
        
        # Import the VASPOutputProcessor from the enhanced module
        # In practice, this would be imported at the top of your main file
        processor = VASPOutputProcessor()
        
        # Ask for processing mode
        print("Processing modes:")
        print("1. Single directory (process one VASP calculation)")
        print("2. Batch mode (process all VASP calculations in a parent directory)")
        
        mode = input("\nSelect mode (1/2) [2]: ").strip() or "2"
        
        if mode == "1":
            # Single directory processing
            calc_dir = input("Path to VASP calculation directory: ").strip()
            if not calc_dir:
                print("Error: No directory provided.")
                return
            
            calc_path = Path(calc_dir).expanduser().resolve()
            if not calc_path.exists() or not calc_path.is_dir():
                print(f"Error: Directory not found: {calc_path}")
                return
            
            try:
                # Process the calculation
                result = processor.process_single_vasp_calculation(calc_path)
                
                # Save JSON file
                output_json = calc_path / f"{calc_path.name}_vasp_output.json"
                with open(output_json, 'w') as f:
                    json.dump(result["optimization_steps"], f, indent=2)
                
                print(f"\nProcessing complete:")
                print(f"  - Optimization steps: {result['num_steps']}")
                print(f"  - Output saved to: {output_json}")
                
            except Exception as e:
                print(f"\nError processing VASP output: {e}")
        
        else:
            # Batch mode processing
            parent_dir = input("Path to parent directory containing VASP calculations: ").strip()
            if not parent_dir:
                print("Error: No directory provided.")
                return
            
            parent_path = Path(parent_dir).expanduser().resolve()
            if not parent_path.exists() or not parent_path.is_dir():
                print(f"Error: Directory not found: {parent_path}")
                return
            
            # Ask for recursive search
            recursive = input("Search recursively for VASP calculations? (y/n) [y]: ").strip().lower() != "n"
            
            try:
                # Process all calculations
                summary = processor.process_directory_batch(parent_path, recursive=recursive)
                
            except Exception as e:
                print(f"\nError during batch processing: {e}")
        
        # Ask if the user wants to process another output
        another = input("\nProcess another output? (y/n): ").strip().lower()
        if another == "y":
            self._handle_output_processing()


#### ML DATASET CREATION METHODS ######

    def _handle_ml_dataset_creation(self):
        """Handle creation of machine learning potential datasets from JSON files."""
        print("\n==== ML Potential Dataset Creation ====\n")
        
        # Step 1: Get JSON file path
        json_path = input("Full path to JSON file containing DFT data (including filename.json): ").strip()
        if not json_path:
            print("Error: No JSON file path provided.")
            return
        
        json_path = Path(json_path).expanduser().resolve()
        
        # Check if path exists and is a file
        if not json_path.exists():
            print(f"Error: File not found: {json_path}")
            return
        if json_path.is_dir():
            print(f"Error: {json_path} is a directory, not a file.")
            print("Please provide the full path to a JSON file.")
            return
        
        # Step 2: Get output directory
        output_dir = input("Output directory for HDF5 datasets [current directory]: ").strip() or "."
        output_dir = Path(output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Step 3: Get dataset name
        default_name = json_path.stem  # Get filename without extension
        dataset_name = input(f"Dataset base name [{default_name}]: ").strip() or default_name
        # Remove .json extension if present in the user input
        if dataset_name.endswith('.json'):
            dataset_name = dataset_name[:-5]
        
        # Step 4: Get train/validation split ratio
        train_ratio = input("Train/validation split ratio [0.85]: ").strip() or "0.85"
        try:
            train_ratio = float(train_ratio)
            if train_ratio <= 0 or train_ratio >= 1:
                raise ValueError("Ratio must be between 0 and 1")
        except ValueError:
            print("Invalid ratio. Using default 0.85")
            train_ratio = 0.85
        
        # Step 5: Get batch size
        batch_size = input("Batch size for HDF5 files [4]: ").strip() or "4"
        try:
            batch_size = int(batch_size)
            if batch_size < 1:
                raise ValueError("Batch size must be at least 1")
        except ValueError:
            print("Invalid batch size. Using default 4")
            batch_size = 4
        
        # Step 6: Get max force threshold
        max_force = input("Maximum force threshold in eV/Å [300.0]: ").strip() or "300.0"
        try:
            max_force = float(max_force)
            if max_force <= 0:
                raise ValueError("Force threshold must be positive")
        except ValueError:
            print("Invalid force threshold. Using default 300.0")
            max_force = 300.0
        
        # Step 7: Check for unit conversion
        units = input("Are energies and forces already in eV units? (y/n) [y]: ").strip().lower() or "y"
        conversion_factor = 1.0
        if units == "n":
            conversion = input("Enter conversion factor to eV [27.2114 for Hartree to eV]: ").strip() or "27.2114"
            try:
                conversion_factor = float(conversion)
            except ValueError:
                print("Invalid conversion factor. Using default 27.2114 (Hartree to eV)")
                conversion_factor = 27.2114
        
        # Step 8: NEW - Periodic Boundary Conditions handling
        print("\n--- Periodic Boundary Conditions (PBC) Handling ---")
        print("Options:")
        print("1. Auto (use PBC settings from input files)")
        print("2. Always use PBC for systems with cell parameters")
        print("3. Never use PBC (treat all systems as isolated)")
        print("4. Custom (specify directions: x, y, z, xy, xz, yz)")
        
        pbc_choice = input("Select PBC handling option (1-4) [1]: ").strip() or "1"
        
        if pbc_choice == "1":
            pbc_handling = "auto"
        elif pbc_choice == "2":
            pbc_handling = "always"
        elif pbc_choice == "3":
            pbc_handling = "never"
        elif pbc_choice == "4":
            print("\nSpecify periodic directions:")
            print("- For 3D periodicity: xyz")
            print("- For 2D periodicity: xy, xz, or yz")
            print("- For 1D periodicity: x, y, or z")
            
            pbc_directions = input("Enter periodic directions [xyz]: ").strip().lower() or "xyz"
            # Validate input (only allow combinations of x, y, z)
            valid_chars = set("xyz")
            if not all(c in valid_chars for c in pbc_directions):
                print("Invalid directions. Using default 'xyz'")
                pbc_handling = "xyz"
            else:
                pbc_handling = pbc_directions
        else:
            print("Invalid choice. Using default 'auto'")
            pbc_handling = "auto"
        
        print("\nCreating ML potential datasets with the following parameters:")
        print(f"  - JSON file: {json_path}")
        print(f"  - Output directory: {output_dir}")
        print(f"  - Dataset name: {dataset_name}")
        print(f"  - Train/validation split: {train_ratio:.2f}/{1-train_ratio:.2f}")
        print(f"  - Batch size: {batch_size}")
        print(f"  - Max force threshold: {max_force} eV/Å")
        print(f"  - Conversion factor: {conversion_factor}")
        print(f"  - PBC handling: {pbc_handling}")
        
        confirm = input("\nProceed with dataset creation? (y/n) [y]: ").strip().lower() or "y"
        if confirm != "y":
            print("Dataset creation canceled.")
            return
        
        print("\nProcessing data and creating HDF5 datasets...")
        try:
            train_h5, valid_h5 = create_mace_h5_dataset(
                json_file=str(json_path),
                output_dir=str(output_dir),
                dataset_name=dataset_name,
                train_ratio=train_ratio,
                batch_size=batch_size,
                max_force_threshold=max_force,
                conversion_factor=conversion_factor,
                pbc_handling=pbc_handling
            )
            print("\nML potential datasets created successfully:")
            print(f"  - Training dataset: {train_h5}")
            print(f"  - Validation dataset: {valid_h5}")
        except Exception as e:
            print(f"\nError creating datasets: {e}")

####### AIMD PROCESSING METHODS ##########

    def _handle_aimd_processing(self):
        """Enhanced AIMD processing with complete multi-file species detection."""
        if not check_ase_available():
            print("\nERROR: ASE (Atomic Simulation Environment) is required for AIMD processing.")
            print("Please install it using: pip install ase\n")
            return
        
        print("\n==== Multi-File Optimized AIMD Processing ====\n")
        
        # Select DFT code
        print("Select DFT code for AIMD:")
        print("1. CP2K (uses existing CP2K parameter system)")
        print("2. VASP (OPTIMIZED with multi-file species detection)")
        
        code_choice = input("\nSelect code (1/2) [2]: ").strip() or "2"
        dft_code = "vasp" if code_choice == "2" else "cp2k"
        print(f"Selected: {dft_code.upper()}")
        
        if dft_code == "vasp":
            print("\n=== MULTI-FILE OPTIMIZED VASP AIMD Configuration ===")
            print("This will:")
            print("1. Analyze ALL JSON files in your directory")
            print("2. Detect species from each file individually")
            print("3. Show you species detected per file")
            print("4. Ask for gamma values for all unique species found")
            print("5. Generate file-specific VASP inputs with correct species order")
            
            # Use the complete optimized function that analyzes all files
            aimd_config = self._get_optimized_vasp_md_parameters_interactively()
            
            if not aimd_config:
                print("Configuration was not completed. Aborting.")
                return
            
            # Get output directory
            output_dir = self._get_output_directory("Enter the output directory for generated files: ")
            if not output_dir:
                return
            
            # Extract the JSON files from the config (they were already analyzed)
            file_species_map = aimd_config.get('FILE_SPECIES_MAP', {})
            if not file_species_map:
                print("ERROR: No file species mapping found. Something went wrong.")
                return
            
            # Get the JSON files that have species (file_species_map now contains Path objects)
            json_files = [file_path for file_path, species_list in file_species_map.items() if species_list]
            
            if not json_files:
                print(f"No JSON files found with detected species.")
                return
            
            print(f"\nProcessing {len(json_files)} JSON files with detected species...")
            
            # Use the optimized processing function
            self._process_json_files_vasp_md_optimized(json_files, output_dir, aimd_config)
            
        else:  # CP2K - keep existing implementation
            print("\n=== CP2K AIMD Configuration ===")
            print("Choose parameter input method:")
            print("1. Template mode (predefined settings)")
            print("2. Interactive mode (custom settings)")
            
            mode = input("\nSelect mode (1/2) [2]: ").strip() or "2"
            
            if mode == "1":
                aimd_config = self._get_aimd_template()
            else:
                aimd_config = self._get_aimd_parameters_interactively()
            
            if not aimd_config:
                print("Configuration was not completed. Aborting.")
                return
            
            # Get paths
            json_path = self._get_json_input_path()
            if not json_path:
                return
            
            output_dir = self._get_output_directory("Enter the output directory for generated files: ")
            if not output_dir:
                return
            
            # Process files
            if json_path.is_file():
                json_files = [json_path]
            else:
                json_files = list(json_path.glob("*.json"))
            
            if not json_files:
                print(f"No JSON files found in {json_path}")
                return
            
            print(f"\nProcessing {len(json_files)} JSON file(s) for CP2K AIMD...")
            self._process_json_files_cp2k_md(json_files, output_dir, aimd_config)

    def _detect_species_from_json(self, json_file: Path) -> List[str]:
        """Extract unique atomic species from JSON file in the correct order."""
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            if not data:
                return []
            
            # Get the final configuration
            final_config = data[-1]
            atom_types = final_config.get("atom_types", [])
            
            if not atom_types:
                return []
            
            # Get unique species in the order they appear (important for VASP)
            from collections import OrderedDict
            unique_species = list(OrderedDict.fromkeys(atom_types))
            
            return unique_species
            
        except Exception as e:
            print(f"Error reading species from {json_file}: {e}")
            return []

    def _analyze_all_json_files_species(self, json_path: Path) -> Dict[Path, List[str]]:
        """Analyze all JSON files and detect species from each one."""
        print(f"Enter path to JSON file or directory: {json_path}")
        
        # Get all JSON files
        if json_path.is_file():
            json_files = [json_path]
        else:
            json_files = list(json_path.glob("*.json"))
            
        if not json_files:
            print(f"No JSON files found in {json_path}")
            return {}
        
        print(f"Found {len(json_files)} JSON files. Analyzing species from each...")
        
        # Dictionary to store file_path -> species mapping
        file_species_map = {}
        all_unique_species = set()
        
        for json_file in json_files:
            print(f"Analyzing species from: {json_file.name}")
            detected_species = self._detect_species_from_json(json_file)
            
            if detected_species:
                file_species_map[json_file] = detected_species
                all_unique_species.update(detected_species)
                print(f"Detected species: {detected_species}")
            else:
                print(f"WARNING: No species detected in {json_file.name}")
                file_species_map[json_file] = []
        
        print(f"\n=== Species Detection Summary ===")
        print(f"Analyzed {len(json_files)} JSON files")
        print(f"Files with detected species: {len([f for f in file_species_map.values() if f])}")
        
        # Show species by file
        print(f"\nSpecies detected per file:")
        for file_path, species in file_species_map.items():
            if species:
                print(f"  {file_path.name}: {species}")
            else:
                print(f"  {file_path.name}: NO SPECIES DETECTED")
        
        # Show all unique species found across all files
        if all_unique_species:
            sorted_species = sorted(list(all_unique_species))
            print(f"\nAll unique species across all files: {sorted_species}")
            print(f"Total unique species: {len(sorted_species)}")
        else:
            print(f"\nERROR: No species detected in any JSON file!")
            return {}
        
        return file_species_map

    def _get_gamma_values_for_all_species(self, file_species_map: Dict[Path, List[str]]) -> Dict[str, float]:
        """Get Langevin gamma values for all unique species found across all JSON files."""
        
        # Get all unique species
        all_species = set()
        for species_list in file_species_map.values():
            all_species.update(species_list)
        
        if not all_species:
            print("No species detected. Cannot configure Langevin parameters.")
            return {}
        
        sorted_species = sorted(list(all_species))
        
        print(f"\n--- Langevin Gamma Configuration ---")
        print(f"You need to specify Langevin friction coefficients (γ) for {len(sorted_species)} unique species:")
        print(f"Species: {', '.join(sorted_species)}")
        print("Typical values: Light atoms (H): 10-20, Medium (C,N,O): 5-15, Heavy: 1-10")
        
        # Option for quick setup
        use_default = input(f"\nUse default γ=10.0 for all {len(sorted_species)} species? (y/n) [n]: ").strip().lower() or "n"
        
        gamma_dict = {}
        
        if use_default == "y":
            for species in sorted_species:
                gamma_dict[species] = 10.0
            print(f"Using γ = 10.0 for all {len(sorted_species)} species")
        else:
            # Collect gamma for each unique species
            print(f"\nEnter gamma values for each species:")
            for species in sorted_species:
                while True:
                    # Show which files contain this species
                    files_with_species = [file_path.name for file_path, species_list in file_species_map.items() 
                                        if species in species_list]
                    files_info = f" (found in {len(files_with_species)} files)"
                    
                    gamma_input = input(f"  Gamma for {species}{files_info} [10.0]: ").strip()
                    try:
                        gamma_val = float(gamma_input) if gamma_input else 10.0
                        if gamma_val <= 0:
                            print("    Error: Gamma must be positive. Try again.")
                            continue
                        gamma_dict[species] = gamma_val
                        print(f"    Set {species}: γ = {gamma_val}")
                        break
                    except ValueError:
                        print("    Error: Invalid number. Try again.")
        
        # Show final configuration
        print(f"\n=== Final Langevin Configuration ===")
        for species, gamma in sorted(gamma_dict.items()):
            files_with_species = [file_path.name for file_path, species_list in file_species_map.items() 
                                if species in species_list]
            print(f"  {species}: γ = {gamma} (used in {len(files_with_species)} files)")
        
        return gamma_dict

    def _get_optimized_vasp_md_parameters_interactively(self) -> Optional[Dict[str, Any]]:
        """Enhanced VASP MD parameter collection with complete multi-file species detection."""
        print("\n==== Optimized VASP MD Parameter Setup ====")
        
        # Step 1: Get JSON directory and analyze ALL files
        print("\n--- Step 1: Species Detection ---")
        print("To optimize the setup, we'll first detect atomic species from your JSON files.")
        
        json_path = self._get_json_input_path()
        if not json_path:
            print("Cannot proceed without JSON path.")
            return None
        
        # Analyze all JSON files for species detection
        file_species_map = self._analyze_all_json_files_species(json_path)
        if not file_species_map:
            print("Species detection failed. Cannot proceed.")
            return None
        
        # Check if any files have species
        files_with_species = {file_path: species for file_path, species in file_species_map.items() if species}
        if not files_with_species:
            print("ERROR: No species detected in any JSON files. Please check file formats.")
            return None
        
        print(f"\nProceeding with {len(files_with_species)} files that have detected species.")
        
        # Step 2: Get base DFT parameters (simplified)
        print("\n--- Step 2: DFT Parameters ---")
        print("Choose DFT parameter method:")
        print("1. Quick defaults (PBE, 400 eV cutoff, optimized for MD)")
        print("2. Custom DFT parameters")
        
        dft_choice = input("Select option (1/2) [1]: ").strip() or "1"
        
        if dft_choice == "2":
            # Use existing comprehensive DFT parameter function
            vasp_config, kpoints_config = self._get_vasp_parameters_interactively()
            if not vasp_config:
                return None
        else:
            # Quick MD-optimized defaults
            vasp_config = {
                "SYSTEM": "Optimized VASP MD calculation",
                "GGA": "PE",          # PBE functional
                "ENCUT": 400,         # Reasonable cutoff
                "EDIFF": 1e-4,        # Relaxed convergence for MD
                "NELM": 40,           # Reduced SCF steps
                "ISMEAR": 0,          # Gaussian smearing
                "SIGMA": 0.05,        # Smearing width
                "PREC": "Normal",     # Normal precision for speed
                "ALGO": "VeryFast",   # Fast algorithm
                "LREAL": "Auto",      # Real space projection for speed
                "LWAVE": False,       # Don't write WAVECAR
                "LCHARG": False,      # Don't write CHGCAR
            }
            kpoints_config = {'type': 'gamma', 'grid': [1, 1, 1]}
            print("Using quick MD-optimized DFT defaults.")
        
        # Step 3: Get MD-specific parameters
        print("\n--- Step 3: MD Parameters ---")
        
        # Temperatures
        temp_input = input("Temperatures in K (comma-separated) [300,400,500]: ").strip() or "300,400,500"
        try:
            temps = [float(t.strip()) for t in temp_input.split(',') if t.strip()]
        except ValueError:
            temps = [300.0, 400.0, 500.0]
            print("Using default temperatures: 300K, 400K, 500K")
        
        vasp_config["temperatures"] = temps
        
        # Ensemble
        print("\nEnsemble selection:")
        print("1. NVT (constant volume and temperature)")
        print("2. NPT (constant pressure and temperature)")
        print("3. NVE (microcanonical)")
        
        ensemble_choice = input("Select ensemble (1/2/3) [1]: ").strip() or "1"
        if ensemble_choice == "2":
            vasp_config["ensemble"] = "NPT"
            vasp_config['ISIF'] = 3
            vasp_config['PSTRESS'] = 0.0
        elif ensemble_choice == "3":
            vasp_config["ensemble"] = "NVE"
        else:
            vasp_config["ensemble"] = "NVT"
        
        # Step 4: Thermostat selection with optimized species handling
        if vasp_config["ensemble"] != "NVE":
            print("\n--- Step 4: Thermostat Selection ---")
            print("1. Nose-Hoover (deterministic)")
            print("2. Langevin (stochastic) - OPTIMIZED for all detected species")
            print("3. Andersen (stochastic)")
            print("4. CSVR (efficient)")
            
            thermo_choice = input("Select thermostat (1/2/3/4) [2]: ").strip() or "2"
            
            if thermo_choice == "2":
                # OPTIMIZED LANGEVIN SETUP FOR ALL FILES
                
                # Get gamma values for all unique species across all files
                gamma_dict = self._get_gamma_values_for_all_species(file_species_map)
                if not gamma_dict:
                    print("Failed to configure Langevin parameters.")
                    return None
                
                # Lattice friction for NPT
                lattice_gamma = 1.0
                if vasp_config["ensemble"] == "NPT":
                    lattice_input = input("\nLattice friction (LANGEVIN_GAMMA_L) [1.0]: ").strip()
                    try:
                        lattice_gamma = float(lattice_input) if lattice_input else 1.0
                    except ValueError:
                        lattice_gamma = 1.0
                
                # Store optimized Langevin parameters
                vasp_config.update({
                    "thermostat": "langevin",
                    "MDALGO": 3,
                    "FILE_SPECIES_MAP": file_species_map,  # Store mapping for later use
                    "GLOBAL_GAMMA_DICT": gamma_dict,       # Global gamma values for all species
                    "LANGEVIN_GAMMA_L": lattice_gamma
                })
                
            else:
                # Handle other thermostats (simplified)
                thermostat_configs = {
                    "1": {"thermostat": "nose_hoover", "MDALGO": 0, "SMASS": 0.5},
                    "3": {"thermostat": "andersen", "MDALGO": 1, "ANDERSEN_PROB": 0.1},
                    "4": {"thermostat": "csvr", "MDALGO": 2, "SMASS": 0.0}
                }
                
                if thermo_choice in thermostat_configs:
                    vasp_config.update(thermostat_configs[thermo_choice])
                else:
                    vasp_config.update(thermostat_configs["1"])  # Default to Nose-Hoover
        
        # Step 5: Basic MD run parameters
        print("\n--- Step 5: MD Run Parameters ---")
        
        try:
            timestep = float(input("Timestep in fs [1.0]: ").strip() or "1.0")
            steps = int(input("Number of MD steps [50000]: ").strip() or "50000")
        except ValueError:
            timestep, steps = 1.0, 50000
            print("Using defaults: 1.0 fs timestep, 50000 steps")
        
        vasp_config.update({
            "IBRION": 0,
            "POTIM": timestep,
            "NSW": steps,
            "TEBEG": temps[0],
            "TEEND": temps[0],
            "NBLOCK": 1,
            "KBLOCK": 10,
            "KPOINTS": kpoints_config
        })
        
        # Final summary
        print(f"\n=== Optimized VASP MD Configuration Summary ===")
        total_files = len(file_species_map)
        files_with_species_count = len([f for f in file_species_map.values() if f])
        print(f"Total JSON files: {total_files}")
        print(f"Files with detected species: {files_with_species_count}")
        
        if vasp_config.get('thermostat') == 'langevin':
            print("Thermostat: Langevin (optimized for all files)")
            gamma_dict = vasp_config.get('GLOBAL_GAMMA_DICT', {})
            for species, gamma in sorted(gamma_dict.items()):
                files_count = len([file_path for file_path, species_list in file_species_map.items() 
                                if species in species_list])
                print(f"  {species}: γ = {gamma} (used in {files_count} files)")
        
        print(f"Temperatures: {', '.join([f'{t}K' for t in temps])}")
        print(f"Ensemble: {vasp_config['ensemble']}")
        print(f"Timestep: {timestep} fs, Steps: {steps}")
        
        confirm = input("\nConfirm configuration? (y/n) [y]: ").strip().lower() or "y"
        return vasp_config if confirm == "y" else None

    def _generate_optimized_vasp_md_incar(self, config: Dict[str, Any], file_specific_species: List[str]) -> str:
        """Generate VASP INCAR with file-specific species-optimized Langevin parameters."""
        lines = []
        
        # System description
        lines.append(f"SYSTEM = {config.get('SYSTEM', 'VASP MD calculation')}")
        lines.append("")
        
        # MD parameters
        lines.append("# MD parameters")
        lines.append(f"IBRION = {config.get('IBRION', 0)}")
        lines.append(f"NSW = {config.get('NSW', 50000)}")
        lines.append(f"POTIM = {config.get('POTIM', 1.0)}")
        lines.append(f"TEBEG = {config.get('TEBEG', 300.0)}")
        lines.append(f"TEEND = {config.get('TEEND', 300.0)}")
        lines.append("ISYM = 0  # Turn off symmetry for MD")
        
        # Thermostat handling
        if 'MDALGO' in config:
            lines.append(f"MDALGO = {config['MDALGO']}")
            
            # OPTIMIZED LANGEVIN HANDLING FOR THIS SPECIFIC FILE
            if config.get('MDALGO') == 3 and 'GLOBAL_GAMMA_DICT' in config:
                global_gamma_dict = config.get('GLOBAL_GAMMA_DICT', {})
                
                # Create gamma list for this file's species in their order
                gamma_values = [global_gamma_dict.get(species, 10.0) for species in file_specific_species]
                gamma_str = ' '.join([f"{val:.1f}" for val in gamma_values])
                
                lines.append(f"LANGEVIN_GAMMA = {gamma_str}")
                lines.append(f"# Species order for this file: {' '.join(file_specific_species)}")
                
                # Add individual gamma values as comment for clarity
                for species in file_specific_species:
                    gamma = global_gamma_dict.get(species, 10.0)
                    lines.append(f"# {species}: γ = {gamma}")
                
                # Lattice friction for NPT
                if 'LANGEVIN_GAMMA_L' in config:
                    lines.append(f"LANGEVIN_GAMMA_L = {config['LANGEVIN_GAMMA_L']}")
                    
            elif config.get('MDALGO') == 1:  # Andersen
                if 'ANDERSEN_PROB' in config:
                    lines.append(f"ANDERSEN_PROB = {config['ANDERSEN_PROB']}")
        
        # Nose-Hoover mass (not for Langevin)
        if 'SMASS' in config and config.get('MDALGO', 0) != 3:
            lines.append(f"SMASS = {config.get('SMASS', 0.5)}")
        
        lines.append(f"ISIF = {config.get('ISIF', 2)}")
        
        if 'PSTRESS' in config:
            lines.append(f"PSTRESS = {config['PSTRESS']}")
        
        lines.append("")
        
        # Rest of the INCAR (electronic structure, XC functional, etc.)
        lines.append("# Electronic structure")
        lines.append(f"ENCUT = {config.get('ENCUT', 400)}")
        lines.append(f"EDIFF = {config.get('EDIFF', 1e-4)}")
        lines.append(f"NELM = {config.get('NELM', 40)}")
        lines.append(f"ISMEAR = {config.get('ISMEAR', 0)}")
        lines.append(f"SIGMA = {config.get('SIGMA', 0.05)}")
        lines.append("")
        
        # XC functional
        lines.append("# Exchange-correlation")
        if 'GGA' in config:
            lines.append(f"GGA = {config['GGA']}")
        if 'METAGGA' in config:
            lines.append(f"METAGGA = {config['METAGGA']}")
        lines.append("")
        
        # Dispersion correction
        if config.get('LVDW', False):
            lines.append("# Dispersion correction")
            lines.append("LVDW = .TRUE.")
            lines.append(f"IVDW = {config.get('IVDW', 11)}")
            if 'VDW_S6' in config:
                lines.append(f"VDW_S6 = {config['VDW_S6']}")
            if 'VDW_R0' in config:
                lines.append(f"VDW_R0 = {config['VDW_R0']}")
            lines.append("")
        
        # Algorithm and precision
        lines.append("# Algorithm and precision")
        lines.append(f"PREC = {config.get('PREC', 'Normal')}")
        lines.append(f"ALGO = {config.get('ALGO', 'VeryFast')}")
        if 'LREAL' in config:
            lines.append(f"LREAL = {config['LREAL']}")
        lines.append("")
        
        # Output control
        lines.append("# Output control")
        lines.append(f"LWAVE = {'.TRUE.' if config.get('LWAVE', False) else '.FALSE.'}")
        lines.append(f"LCHARG = {'.TRUE.' if config.get('LCHARG', False) else '.FALSE.'}")
        lines.append(f"NBLOCK = {config.get('NBLOCK', 1)}")
        lines.append(f"KBLOCK = {config.get('KBLOCK', 10)}")
        lines.append("LCHIMAG = .FALSE.")
        lines.append("")
        
        return "\n".join(lines)

    def _process_json_files_vasp_md_optimized(self, json_files: List[Path], output_dir: Path, config: Dict[str, Any]):
        """Process JSON files with file-specific species-optimized Langevin parameters."""
        temperatures = config.get('temperatures', [300.0])
        processed_count = 0
        
        # Extract data from config
        kpoints_config = config.pop('KPOINTS', {'type': 'gamma', 'grid': [1, 1, 1]})
        file_species_map = config.get('FILE_SPECIES_MAP', {})
        global_gamma_dict = config.get('GLOBAL_GAMMA_DICT', {})
        
        print(f"\nProcessing {len(json_files)} files with optimized species-specific parameters...")
        print(f"Global gamma configuration: {global_gamma_dict}")
        
        for json_file in json_files:
            try:
                print(f"\nProcessing {json_file.name}...", end=" ")
                
                # Get species for this specific file
                file_species = file_species_map.get(json_file, [])
                if not file_species:
                    print("ERROR: No species detected for this file")
                    continue
                
                print(f"Species: {file_species}")
                
                # Extract structure
                atoms = extract_final_structure_from_json(json_file)
                if atoms is None:
                    print("ERROR: Could not extract structure")
                    continue
                
                # Generate input files for each temperature
                base_name = json_file.stem
                success_count = 0
                
                for temp in temperatures:
                    temp_dir = output_dir / base_name / f"T{int(temp)}K"
                    
                    # Update config for this temperature and file
                    temp_config = config.copy()
                    temp_config['TEBEG'] = temp
                    temp_config['TEEND'] = temp
                    temp_config['SYSTEM'] = f"{base_name} MD at {int(temp)}K"
                    
                    # Write VASP MD input files with file-specific species
                    if self._write_optimized_vasp_md_input(atoms, temp_dir, temp_config, kpoints_config, file_species):
                        success_count += 1
                    else:
                        print(f"ERROR: Failed to write files for {temp}K")
                        break
                
                if success_count == len(temperatures):
                    processed_count += 1
                    print(f"OK - Generated {success_count} optimized temperature setups")
                else:
                    print(f"PARTIAL - Generated {success_count}/{len(temperatures)} setups")
                    
            except Exception as e:
                print(f"ERROR: {str(e)}")
        
        print(f"\nProcessing Summary:")
        print(f"Successfully processed: {processed_count} out of {len(json_files)} JSON files")
        print(f"Total input sets generated: {processed_count * len(temperatures)}")
        print(f"All files use optimized species-specific Langevin parameters")

    def _write_optimized_vasp_md_input(self, atoms, output_dir: Path, config: Dict[str, Any], 
                                    kpoints_config: Dict[str, Any], file_species: List[str]) -> bool:
        """Write optimized VASP MD input files with file-specific species."""
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate optimized INCAR with file-specific species
            incar_content = self._generate_optimized_vasp_md_incar(config, file_species)
            
            # Generate POSCAR
            poscar_content = self._generate_poscar_from_atoms(atoms, config['SYSTEM'])
            
            # Generate KPOINTS
            kpoints_content = self._generate_vasp_kpoints(kpoints_config)
            
            # Write files
            (output_dir / "INCAR").write_text(incar_content)
            (output_dir / "POSCAR").write_text(poscar_content)
            (output_dir / "KPOINTS").write_text(kpoints_content)
            
            # Write run script
            run_script = self._generate_vasp_md_run_script(config)
            (output_dir / "run_vasp.sh").write_text(run_script)
            (output_dir / "run_vasp.sh").chmod(0o755)
            
            return True
            
        except Exception as e:
            print(f"Error writing optimized VASP MD files: {e}")
            return False

    def _get_langevin_gamma_for_species(self, species_list: List[str]) -> Dict[str, float]:
        """Get Langevin gamma values for each detected species."""
        if not species_list:
            print("No species detected. Using default configuration.")
            return {"Unknown": 10.0}
        
        print(f"\nDetected atomic species: {', '.join(species_list)}")
        print("Please specify Langevin friction coefficients (γ) for each species:")
        print("Typical values: Light atoms (H): 10-20, Medium (C,N,O): 5-15, Heavy: 1-10")
        
        # Option for quick setup
        use_default = input(f"\nUse default γ=10.0 for all {len(species_list)} species? (y/n) [n]: ").strip().lower() or "n"
        
        gamma_dict = {}
        
        if use_default == "y":
            for species in species_list:
                gamma_dict[species] = 10.0
            print("Using γ = 10.0 for all species")
        else:
            # Collect gamma for each species
            for species in species_list:
                while True:
                    gamma_input = input(f"  Gamma for {species} [10.0]: ").strip()
                    try:
                        gamma_val = float(gamma_input) if gamma_input else 10.0
                        if gamma_val <= 0:
                            print("    Error: Gamma must be positive. Try again.")
                            continue
                        gamma_dict[species] = gamma_val
                        print(f"    Set {species}: γ = {gamma_val}")
                        break
                    except ValueError:
                        print("    Error: Invalid number. Try again.")
        
        return gamma_dict

    def _get_vasp_thermostat_parameters(self) -> Dict[str, Any]:
        """Get thermostat-specific parameters for VASP MD."""
        print("\n--- Thermostat Selection ---")
        print("1. Nose-Hoover (deterministic, good for equilibrium properties)")
        print("2. Langevin (stochastic, good for thermalization)")
        print("3. Andersen (stochastic, simple)")
        print("4. CSVR (efficient, good compromise)")
        print("5. Nose-Hoover Chain (improved Nose-Hoover)")
        
        choice = input("Select thermostat (1-5) [1]: ").strip() or "1"
        
        if choice == "2":
            print("\n--- Langevin Thermostat Parameters ---")
            print("VASP requires one LANGEVIN_GAMMA value for each atomic species in your system.")
            print("The order should match the order of species in your POSCAR file.")
            
            # Ask for number of species
            num_species = input("Number of atomic species in your system: ").strip()
            try:
                num_species = int(num_species)
                if num_species < 1:
                    raise ValueError("Must have at least 1 species")
            except ValueError:
                print("Invalid input. Please enter a positive integer.")
                # Recursively call to get valid input
                return self._get_vasp_thermostat_parameters()
            
            # Ask for species names and gamma values
            gamma_values = []
            species_names = []
            
            print(f"\nEnter information for {num_species} species:")
            for i in range(num_species):
                print(f"\nSpecies {i+1}:")
                species_name = input(f"  Element symbol (e.g., H, C, O): ").strip()
                if not species_name:
                    species_name = f"Species{i+1}"
                species_names.append(species_name)
                
                gamma_input = input(f"  Langevin friction coefficient for {species_name} [10.0]: ").strip()
                try:
                    gamma_val = float(gamma_input) if gamma_input else 10.0
                    if gamma_val < 0:
                        print("  Warning: Negative gamma value. Using 10.0 instead.")
                        gamma_val = 10.0
                except ValueError:
                    print("  Invalid gamma value. Using default 10.0.")
                    gamma_val = 10.0
                
                gamma_values.append(gamma_val)
                print(f"  Set {species_name}: γ = {gamma_val}")
            
            # Show summary
            print(f"\nLangevin parameters summary:")
            for species, gamma in zip(species_names, gamma_values):
                print(f"  {species}: γ = {gamma}")
            
            # Ask for lattice friction for NPT
            lattice_gamma = input("\nLattice friction coefficient (LANGEVIN_GAMMA_L) for NPT [1.0]: ").strip()
            try:
                lattice_gamma_val = float(lattice_gamma) if lattice_gamma else 1.0
            except ValueError:
                lattice_gamma_val = 1.0
            
            return {
                "thermostat": "langevin",
                "MDALGO": 3,  # Langevin
                "LANGEVIN_GAMMA_LIST": gamma_values,  # List of gamma values per species
                "ATOMIC_SPECIES": species_names,      # List of species names
                "LANGEVIN_GAMMA_L": lattice_gamma_val  # Lattice friction
            }
            
        elif choice == "3":
            print("\n--- Andersen Thermostat Parameters ---")
            prob = input("Andersen collision probability [0.1]: ").strip()
            try:
                prob_val = float(prob) if prob else 0.1
            except ValueError:
                prob_val = 0.1
            
            return {
                "thermostat": "andersen", 
                "MDALGO": 1,  # Andersen
                "ANDERSEN_PROB": prob_val
            }
        
        elif choice == "4":
            print("\n--- CSVR Thermostat Parameters ---")
            return {
                "thermostat": "csvr",
                "MDALGO": 2,  # CSVR
                "SMASS": 0.0  # Not used for CSVR but set to 0
            }
        
        elif choice == "5":
            print("\n--- Nose-Hoover Chain Parameters ---")
            chains = input("Number of chains [4]: ").strip()
            try:
                chains_val = int(chains) if chains else 4
            except ValueError:
                chains_val = 4
                
            return {
                "thermostat": "nhc",
                "MDALGO": 0,  # NHC is default
                "SMASS": 0.5,  # Nose mass parameter
                "NBLOCK": 1   # For better energy conservation
            }
        
        else:
            print("\n--- Nose-Hoover Thermostat Parameters ---")
            smass = input("Nose mass parameter (SMASS) [0.5]: ").strip()
            try:
                smass_val = float(smass) if smass else 0.5
            except ValueError:
                smass_val = 0.5
            
            return {
                "thermostat": "nose_hoover",
                "MDALGO": 0,  # Standard Nose-Hoover
                "SMASS": smass_val
            }

    def _generate_enhanced_vasp_md_incar(self, config: Dict[str, Any]) -> str:
        """Generate enhanced VASP INCAR for MD - FIXED LANGEVIN VERSION."""
        lines = []
        
        # System description
        lines.append(f"SYSTEM = {config.get('SYSTEM', 'VASP MD calculation')}")
        lines.append("")
        
        # MD parameters (always first section)
        lines.append("# MD parameters")
        lines.append(f"IBRION = {config.get('IBRION', 0)}")
        lines.append(f"NSW = {config.get('NSW', 50000)}")
        lines.append(f"POTIM = {config.get('POTIM', 1.0)}")
        lines.append(f"TEBEG = {config.get('TEBEG', 300.0)}")
        lines.append(f"TEEND = {config.get('TEEND', 300.0)}")
        
        # CRITICAL: Turn off symmetry for MD (fixes the WARNING)
        lines.append("ISYM = 0")
        
        # Handle different thermostats
        if 'MDALGO' in config:
            lines.append(f"MDALGO = {config['MDALGO']}")
            
            # FIXED: Proper handling of Langevin thermostat
            if config.get('MDALGO') == 3:  # Langevin
                if 'LANGEVIN_GAMMA_LIST' in config:
                    gamma_values = config['LANGEVIN_GAMMA_LIST']
                    atomic_species = config.get('ATOMIC_SPECIES', [])
                    
                    # Format: LANGEVIN_GAMMA = value1 value2 value3 ...
                    gamma_str = ' '.join([f"{val:.1f}" for val in gamma_values])
                    lines.append(f"LANGEVIN_GAMMA = {gamma_str}")
                    
                    # Add comment for clarity
                    if atomic_species:
                        species_str = ' '.join(atomic_species)
                        lines.append(f"# LANGEVIN_GAMMA values correspond to: {species_str}")
                    
                    # Lattice friction (for NPT)
                    if 'LANGEVIN_GAMMA_L' in config:
                        lines.append(f"LANGEVIN_GAMMA_L = {config['LANGEVIN_GAMMA_L']}")
                else:
                    # This should not happen with the fixed function, but keep as fallback
                    print("WARNING: LANGEVIN_GAMMA_LIST not found. Using fallback values.")
                    gamma_base = config.get('LANGEVIN_GAMMA_BASE', 10.0)
                    lines.append(f"LANGEVIN_GAMMA = {gamma_base:.1f}")
                    lines.append("# WARNING: Single gamma value used - should specify per species")
                    
            elif config.get('MDALGO') == 1:  # Andersen
                if 'ANDERSEN_PROB' in config:
                    lines.append(f"ANDERSEN_PROB = {config['ANDERSEN_PROB']}")
        
        # Nose-Hoover mass parameter (for MDALGO=0 or if SMASS is specified)
        if 'SMASS' in config and config.get('MDALGO', 0) != 3:  # Don't use SMASS with Langevin
            lines.append(f"SMASS = {config.get('SMASS', 0.5)}")
        
        lines.append(f"ISIF = {config.get('ISIF', 2)}")
        
        if 'PSTRESS' in config:
            lines.append(f"PSTRESS = {config['PSTRESS']}")
        
        lines.append("")
        
        # Electronic structure parameters
        lines.append("# Electronic structure")
        lines.append(f"ENCUT = {config.get('ENCUT', 400)}")
        lines.append(f"EDIFF = {config.get('EDIFF', 1e-4)}")
        lines.append(f"NELM = {config.get('NELM', 40)}")
        lines.append(f"ISMEAR = {config.get('ISMEAR', 0)}")
        lines.append(f"SIGMA = {config.get('SIGMA', 0.05)}")
        lines.append("")
        
        # XC functional
        lines.append("# Exchange-correlation")
        if 'GGA' in config:
            lines.append(f"GGA = {config['GGA']}")
        if 'METAGGA' in config:
            lines.append(f"METAGGA = {config['METAGGA']}")
        lines.append("")
        
        # Dispersion correction
        if config.get('LVDW', False):
            lines.append("# Dispersion correction")
            lines.append("LVDW = .TRUE.")
            lines.append(f"IVDW = {config.get('IVDW', 11)}")
            if 'VDW_S6' in config:
                lines.append(f"VDW_S6 = {config['VDW_S6']}")
            if 'VDW_R0' in config:
                lines.append(f"VDW_R0 = {config['VDW_R0']}")
            lines.append("")
        
        # Algorithm and precision
        lines.append("# Algorithm and precision")
        lines.append(f"PREC = {config.get('PREC', 'Normal')}")
        lines.append(f"ALGO = {config.get('ALGO', 'VeryFast')}")
        if 'LREAL' in config:
            lines.append(f"LREAL = {config['LREAL']}")
        lines.append("")
        
        # Output control
        lines.append("# Output control")
        lines.append(f"LWAVE = {'.TRUE.' if config.get('LWAVE', False) else '.FALSE.'}")
        lines.append(f"LCHARG = {'.TRUE.' if config.get('LCHARG', False) else '.FALSE.'}")
        lines.append(f"NBLOCK = {config.get('NBLOCK', 1)}")
        lines.append(f"KBLOCK = {config.get('KBLOCK', 10)}")
        lines.append("")
        
        # MD-specific output settings
        lines.append("# MD output settings")
        lines.append("LCHIMAG = .FALSE.")  # Don't write imaginary part
        lines.append("")
        
        # Magnetic properties if specified
        if config.get('ISPIN', 1) == 2:
            lines.append("# Magnetic properties")
            lines.append("ISPIN = 2")
            if 'MAGMOM' in config:
                lines.append(f"MAGMOM = {config['MAGMOM']}")
            lines.append("")
        
        return "\n".join(lines)

    # Optional: Add a helper function to auto-detect species from structure
    def _detect_species_from_atoms(self, atoms):
        """Detect unique atomic species from ASE atoms object."""
        if atoms is None:
            return []
        
        from collections import OrderedDict
        
        # Get unique species in the order they appear
        species = list(OrderedDict.fromkeys(atoms.get_chemical_symbols()))
        return species

    def _auto_configure_langevin_gamma(self, atoms, default_gamma=10.0):
        """Auto-configure Langevin gamma values for all species in the system."""
        species = self._detect_species_from_atoms(atoms)
        
        if not species:
            print("Warning: Could not detect atomic species. Using default configuration.")
            return {
                "LANGEVIN_GAMMA_LIST": [default_gamma],
                "ATOMIC_SPECIES": ["Unknown"]
            }
        
        print(f"\nDetected atomic species: {', '.join(species)}")
        use_auto = input(f"Use default gamma={default_gamma} for all species? (y/n) [y]: ").strip().lower() or "y"
        
        if use_auto == "y":
            gamma_values = [default_gamma] * len(species)
            print(f"Using γ = {default_gamma} for all {len(species)} species")
        else:
            gamma_values = []
            print("Enter custom gamma values:")
            for species_name in species:
                gamma_input = input(f"  Gamma for {species_name} [{default_gamma}]: ").strip()
                try:
                    gamma_val = float(gamma_input) if gamma_input else default_gamma
                except ValueError:
                    gamma_val = default_gamma
                gamma_values.append(gamma_val)
        
        return {
            "LANGEVIN_GAMMA_LIST": gamma_values,
            "ATOMIC_SPECIES": species
        }

    def _get_vasp_md_parameters_interactively(self) -> Optional[Dict[str, Any]]:
        """Enhanced VASP MD parameter collection using existing DFT parameter selection."""
        print("\n==== Enhanced VASP MD Parameter Setup ====")
        
        # Step 1: Get comprehensive DFT parameters using existing function
        print("\n--- Step 1: DFT Parameters ---")
        print("First, let's configure the DFT calculation parameters...")
        
        vasp_config, kpoints_config = self._get_vasp_parameters_interactively()
        if not vasp_config:
            print("DFT parameter configuration was cancelled.")
            return None
        
        # Step 2: Modify the calculation type to MD and remove incompatible parameters
        print("\n--- Step 2: Converting to MD Calculation ---")
        
        # Remove optimization-specific parameters
        md_incompatible_keys = ['IBRION', 'NSW', 'ISIF', 'EDIFFG']
        for key in md_incompatible_keys:
            if key in vasp_config:
                del vasp_config[key]
        
        # Set MD-specific basic parameters
        vasp_config['IBRION'] = 0  # MD
        vasp_config['NSW'] = 50000  # Will be updated based on user input
        vasp_config['ISIF'] = 2    # Stress tensor calculation
        
        # Step 3: Get MD-specific parameters
        print("\n--- Step 3: MD-Specific Parameters ---")
        
        # Get temperatures
        temp_input = input("\nEnter temperatures in K (comma-separated, e.g., 200,300,400): ").strip()
        try:
            temps = [float(t.strip()) for t in temp_input.split(',') if t.strip()]
            if not temps:
                temps = [300.0]
            vasp_config["temperatures"] = temps
        except ValueError:
            print("Invalid temperature format. Using 300K.")
            vasp_config["temperatures"] = [300.0]
        
        # Get ensemble
        print("\n--- Ensemble Selection ---")
        print("1. NVT (constant volume and temperature)")
        print("2. NPT (constant pressure and temperature)") 
        print("3. NVE (microcanonical - constant energy)")
        
        ensemble_choice = input("Select ensemble (1/2/3) [1]: ").strip() or "1"
        if ensemble_choice == "2":
            vasp_config["ensemble"] = "NPT"
            # Add NPT-specific parameters
            vasp_config['ISIF'] = 3  # Full stress tensor for NPT
            vasp_config['PSTRESS'] = 0.0  # External pressure
        elif ensemble_choice == "3":
            vasp_config["ensemble"] = "NVE"
        else:
            vasp_config["ensemble"] = "NVT"
        
        # Get thermostat (if not NVE)
        if vasp_config["ensemble"] != "NVE":
            thermostat_params = self._get_vasp_thermostat_parameters()
            vasp_config.update(thermostat_params)
        
        # Get MD run parameters
        print("\n--- MD Run Parameters ---")
        
        try:
            timestep = float(input("Timestep in fs [1.0]: ").strip() or "1.0")
            steps = int(input("Number of MD steps [50000]: ").strip() or "50000")
            vasp_config.update({
                "POTIM": timestep,
                "NSW": steps
            })
        except ValueError:
            print("Invalid input. Using defaults (1.0 fs, 50000 steps).")
            vasp_config.update({"POTIM": 1.0, "NSW": 50000})
        
        # Set temperature parameters
        if vasp_config["ensemble"] in ["NVT", "NPT"]:
            # Use the first temperature as default, but all will be used for file generation
            first_temp = vasp_config["temperatures"][0]
            vasp_config["TEBEG"] = first_temp
            vasp_config["TEEND"] = first_temp
        
        # NPT pressure settings
        if vasp_config["ensemble"] == "NPT":
            try:
                pressure = float(input("Target pressure in kBar [0.0]: ").strip() or "0.0")
                vasp_config["PSTRESS"] = pressure
            except ValueError:
                vasp_config["PSTRESS"] = 0.0
        
        # Step 4: Optimize settings for MD
        print("\n--- Step 4: MD Optimization ---")
        
        # Suggest optimizations for MD
        optimize = input("Apply MD-optimized settings? (y/n) [y]: ").strip().lower() or "y"
        if optimize == "y":
            # Reduce precision for speed in MD
            if vasp_config.get('PREC') == 'Accurate':
                vasp_config['PREC'] = 'Normal'
                print("  - Reduced PREC to Normal for MD efficiency")
            
            # Use faster algorithm for MD
            if vasp_config.get('ALGO') == 'Normal':
                vasp_config['ALGO'] = 'VeryFast'
                print("  - Changed ALGO to VeryFast for MD efficiency")
            
            # Optimize SCF settings for MD
            if vasp_config.get('NELM', 60) > 40:
                vasp_config['NELM'] = 40
                print("  - Reduced NELM to 40 for MD efficiency")
            
            # Loosen SCF convergence slightly for MD
            if vasp_config.get('EDIFF', 1e-5) < 1e-4:
                vasp_config['EDIFF'] = 1e-4
                print("  - Adjusted EDIFF to 1e-4 for MD efficiency")
            
            # Set LREAL for larger systems
            if 'LREAL' not in vasp_config:
                vasp_config['LREAL'] = 'Auto'
                print("  - Set LREAL to Auto for MD efficiency")
        
        # Step 5: Output settings for MD
        print("\n--- Step 5: MD Output Settings ---")
        
        # Always write XDATCAR for MD trajectories
        vasp_config['LWAVE'] = False  # Don't need WAVECAR for MD
        vasp_config['LCHARG'] = False  # Don't need CHGCAR for MD
        
        # MD-specific output settings
        try:
            nblock = int(input("Write output every N steps (NBLOCK) [1]: ").strip() or "1")
            kblock = int(input("Write XDATCAR every N*NBLOCK steps (KBLOCK) [10]: ").strip() or "10")
            vasp_config.update({"NBLOCK": nblock, "KBLOCK": kblock})
        except ValueError:
            vasp_config.update({"NBLOCK": 1, "KBLOCK": 10})
        
        # Store k-points configuration
        vasp_config['KPOINTS'] = kpoints_config
        
        # Step 6: Final summary
        print(f"\n=== Enhanced VASP MD Configuration Summary ===")
        print(f"System Description: {vasp_config.get('SYSTEM', 'VASP MD calculation')}")
        print(f"Temperatures: {', '.join([f'{t}K' for t in vasp_config['temperatures']])}")
        print(f"Ensemble: {vasp_config['ensemble']}")
        
        if vasp_config['ensemble'] != 'NVE':
            print(f"Thermostat: {vasp_config.get('thermostat', 'nose_hoover')}")
        
        print(f"Timestep: {vasp_config['POTIM']} fs")
        print(f"MD Steps: {vasp_config['NSW']}")
        print(f"XC Functional: {vasp_config.get('GGA', 'PBE') if 'GGA' in vasp_config else 'LDA'}")
        
        if vasp_config.get('LVDW', False):
            print(f"Dispersion Correction: IVDW={vasp_config.get('IVDW', 0)}")
        
        print(f"Plane Wave Cutoff: {vasp_config.get('ENCUT', 400)} eV")
        print(f"K-point Grid: {kpoints_config.get('grid', [1,1,1])}")
        print(f"Electronic Smearing: ISMEAR={vasp_config.get('ISMEAR', 0)}, SIGMA={vasp_config.get('SIGMA', 0.05)}")
        
        if vasp_config['ensemble'] == 'NPT':
            print(f"Target Pressure: {vasp_config.get('PSTRESS', 0.0)} kBar")
        
        print(f"Output Frequency: Every {vasp_config.get('NBLOCK', 1)} steps")
        print(f"Trajectory Frequency: Every {vasp_config.get('KBLOCK', 10)*vasp_config.get('NBLOCK', 1)} steps")
        
        confirm = input("\nConfirm these settings? (y/n) [y]: ").strip().lower() or "y"
        return vasp_config if confirm == "y" else None

    def _select_vasp_md_template(self) -> Optional[Dict[str, Any]]:
        """Enhanced VASP MD template selection using existing DFT parameters."""
        print("\n==== Enhanced VASP MD Template Selection ====")
        print("\nThis will first set up DFT parameters, then apply MD templates...")
        
        # Step 1: Get base DFT configuration
        print("\n--- Step 1: Base DFT Configuration ---")
        print("Choose base DFT settings:")
        print("1. Use existing VASP template and customize for MD")
        print("2. Use quick defaults optimized for MD")
        
        base_choice = input("Select option (1/2) [2]: ").strip() or "2"
        
        if base_choice == "1":
            # Use existing VASP template system
            base_config = self._get_vasp_template()
            if not base_config:
                return None
            
            # Extract kpoints if present
            kpoints_config = base_config.pop('KPOINTS', {'type': 'gamma', 'grid': [1, 1, 1]})
        else:
            # Use MD-optimized defaults
            base_config = {
                "SYSTEM": "VASP MD calculation",
                "ENCUT": 400,
                "EDIFF": 1e-4,  # Slightly relaxed for MD
                "NELM": 40,     # Reduced for MD efficiency
                "ISMEAR": 0,
                "SIGMA": 0.05,
                "PREC": "Normal",  # Normal precision for MD
                "ALGO": "VeryFast",  # Fast algorithm for MD
                "LREAL": "Auto",   # For efficiency
                "LWAVE": False,
                "LCHARG": False
            }
            kpoints_config = {'type': 'gamma', 'grid': [1, 1, 1]}  # Gamma point for MD
            
            print("Using MD-optimized default DFT settings.")
        
        # Step 2: Select MD template
        print("\n--- Step 2: MD Template Selection ---")
        print("\nAvailable MD templates:")
        print("1. Standard NVT (Nose-Hoover, 300K)")
        print("2. Multi-temperature NVT (200K, 300K, 400K, 500K)")
        print("3. NPT ensemble (1 bar, 300K)")
        print("4. NVE ensemble (microcanonical)")
        print("5. High-temperature study (800K, 1000K, 1200K)")
        print("6. Liquid-state MD (400K, 600K, 800K)")
        
        choice = input("\nSelect MD template (1-6) [1]: ").strip() or "1"
        
        # MD template configurations
        md_templates = {
            "1": {
                "name": "Standard NVT",
                "temperatures": [300.0],
                "ensemble": "NVT",
                "MDALGO": 0,      # Nose-Hoover
                "SMASS": 0.5,
                "POTIM": 1.0,
                "NSW": 50000,
                "NBLOCK": 1,
                "KBLOCK": 10
            },
            "2": {
                "name": "Multi-temperature NVT",
                "temperatures": [200.0, 300.0, 400.0, 500.0],
                "ensemble": "NVT",
                "MDALGO": 0,
                "SMASS": 0.5,
                "POTIM": 1.0,
                "NSW": 50000,
                "NBLOCK": 1,
                "KBLOCK": 10
            },
            "3": {
                "name": "NPT ensemble",
                "temperatures": [300.0],
                "ensemble": "NPT",
                "MDALGO": 0,
                "SMASS": 0.5,
                "POTIM": 1.0,
                "NSW": 50000,
                "ISIF": 3,        # Full stress tensor
                "PSTRESS": 0.0,   # 1 bar = ~0.001 kBar
                "NBLOCK": 1,
                "KBLOCK": 10
            },
            "4": {
                "name": "NVE ensemble",
                "temperatures": [300.0],  # Initial temperature
                "ensemble": "NVE",
                "MDALGO": 0,
                "SMASS": -1,      # No thermostat
                "POTIM": 0.5,     # Smaller timestep for stability
                "NSW": 50000,
                "NBLOCK": 1,
                "KBLOCK": 5       # More frequent output for NVE
            },
            "5": {
                "name": "High-temperature study",
                "temperatures": [800.0, 1000.0, 1200.0],
                "ensemble": "NVT",
                "MDALGO": 3,      # Langevin for high T
                "LANGEVIN_GAMMA": [10.0],
                "POTIM": 0.5,     # Smaller timestep for stability
                "NSW": 30000,     # Shorter runs at high T
                "NBLOCK": 1,
                "KBLOCK": 5
            },
            "6": {
                "name": "Liquid-state MD",
                "temperatures": [400.0, 600.0, 800.0],
                "ensemble": "NVT", 
                "MDALGO": 3,      # Langevin
                "LANGEVIN_GAMMA": [5.0],
                "POTIM": 1.0,
                "NSW": 100000,    # Longer runs for liquids
                "NBLOCK": 2,      # Less frequent output
                "KBLOCK": 20
            }
        }
        
        if choice not in md_templates:
            choice = "1"
        
        md_config = md_templates[choice].copy()
        
        # Step 3: Merge configurations
        # Start with base DFT config, then add MD parameters
        combined_config = base_config.copy()
        combined_config.update(md_config)
        
        # Ensure MD-specific parameters override any conflicts
        combined_config.update({
            "IBRION": 0,      # MD
            "ISIF": md_config.get("ISIF", 2),
            "TEBEG": md_config["temperatures"][0],
            "TEEND": md_config["temperatures"][0]
        })
        
        # Add KPOINTS configuration
        combined_config['KPOINTS'] = kpoints_config
        
        # Step 4: Show template details and allow customization
        print(f"\nSelected: {md_config['name']}")
        print(f"Temperatures: {', '.join([f'{t}K' for t in md_config['temperatures']])}")
        print(f"Ensemble: {md_config['ensemble']}")
        print(f"Timestep: {md_config['POTIM']} fs")
        print(f"Steps: {md_config['NSW']}")
        
        if md_config['ensemble'] != 'NVE':
            if combined_config.get('MDALGO') == 0:
                print(f"Thermostat: Nose-Hoover (SMASS={combined_config.get('SMASS', 0.5)})")
            elif combined_config.get('MDALGO') == 3:
                print(f"Thermostat: Langevin (γ={combined_config.get('LANGEVIN_GAMMA', [10.0])[0]})")
        
        # Option to customize
        customize = input("\nCustomize this template? (y/n) [n]: ").strip().lower() or "n"
        if customize == "y":
            # Allow temperature customization
            temp_input = input(f"Custom temperatures (comma-separated) [{','.join(map(str, md_config['temperatures']))}]: ").strip()
            if temp_input:
                try:
                    custom_temps = [float(t.strip()) for t in temp_input.split(',') if t.strip()]
                    if custom_temps:
                        combined_config["temperatures"] = custom_temps
                        combined_config["TEBEG"] = custom_temps[0]
                        combined_config["TEEND"] = custom_temps[0]
                except ValueError:
                    print("Invalid temperature format. Using template defaults.")
            
            # Allow timestep customization
            timestep_input = input(f"Custom timestep in fs [{md_config['POTIM']}]: ").strip()
            if timestep_input:
                try:
                    custom_timestep = float(timestep_input)
                    combined_config["POTIM"] = custom_timestep
                except ValueError:
                    print("Invalid timestep. Using template default.")
            
            # Allow steps customization
            steps_input = input(f"Custom number of steps [{md_config['NSW']}]: ").strip()
            if steps_input:
                try:
                    custom_steps = int(steps_input)
                    combined_config["NSW"] = custom_steps
                except ValueError:
                    print("Invalid steps. Using template default.")
        
        # Final confirmation
        print(f"\n=== Final Template Configuration ===")
        print(f"Template: {md_config['name']}")
        print(f"Temperatures: {', '.join([f'{t}K' for t in combined_config['temperatures']])}")
        print(f"DFT Method: {combined_config.get('GGA', 'LDA')} with ENCUT={combined_config.get('ENCUT', 400)} eV")
        print(f"MD: {combined_config['ensemble']} ensemble, {combined_config['POTIM']} fs timestep, {combined_config['NSW']} steps")
        
        confirm = input("\nUse this template? (y/n) [y]: ").strip().lower() or "y"
        return combined_config if confirm == "y" else None

    def _get_quick_vasp_md_defaults(self) -> Dict[str, Any]:
        """Quick VASP MD setup with sensible defaults."""
        print("\n==== Quick VASP MD Setup ====")
        
        # Get just the essential parameters
        temp_input = input("Temperatures in K (comma-separated) [300,400,500]: ").strip() or "300,400,500"
        try:
            temps = [float(t.strip()) for t in temp_input.split(',') if t.strip()]
        except ValueError:
            temps = [300.0, 400.0, 500.0]
            print("Using default temperatures: 300K, 400K, 500K")
        
        ensemble = input("Ensemble (NVT/NPT) [NVT]: ").strip().upper() or "NVT"
        if ensemble not in ["NVT", "NPT"]:
            ensemble = "NVT"
        
        # MD-optimized defaults
        config = {
            "SYSTEM": "Quick VASP MD calculation",
            "temperatures": temps,
            "ensemble": ensemble,
            
            # DFT settings optimized for MD
            "GGA": "PE",          # PBE functional
            "ENCUT": 400,         # Reasonable cutoff
            "EDIFF": 1e-4,        # Relaxed convergence for MD
            "NELM": 40,           # Reduced SCF steps
            "ISMEAR": 0,          # Gaussian smearing
            "SIGMA": 0.05,        # Smearing width
            "PREC": "Normal",     # Normal precision for speed
            "ALGO": "VeryFast",   # Fast algorithm
            "LREAL": "Auto",      # Real space projection for speed
            "LWAVE": False,       # Don't write WAVECAR
            "LCHARG": False,      # Don't write CHGCAR
            
            # MD settings
            "IBRION": 0,          # MD
            "POTIM": 1.0,         # 1 fs timestep
            "NSW": 50000,         # 50 ps simulation
            "ISIF": 3 if ensemble == "NPT" else 2,
            "MDALGO": 0,          # Nose-Hoover
            "SMASS": 0.5,         # Nose mass
            "TEBEG": temps[0],    # Start temperature
            "TEEND": temps[0],    # End temperature (will be set per temperature)
            
            # Output settings
            "NBLOCK": 1,          # Output every step
            "KBLOCK": 10,         # XDATCAR every 10 steps
            
            # K-points (Gamma point for MD)
            "KPOINTS": {"type": "gamma", "grid": [1, 1, 1]}
        }
        
        if ensemble == "NPT":
            config["PSTRESS"] = 0.0  # 1 bar pressure
        
        print(f"Quick setup complete:")
        print(f"  - Temperatures: {', '.join(map(str, temps))}K")
        print(f"  - Ensemble: {ensemble}")
        print(f"  - Functional: PBE")
        print(f"  - Timestep: 1.0 fs")
        print(f"  - Steps: 50000 (50 ps)")
        
        confirm = input("\nUse these quick defaults? (y/n) [y]: ").strip().lower() or "y"
        return config if confirm == "y" else None

    def _process_json_files_vasp_md_enhanced(self, json_files: List[Path], output_dir: Path, config: Dict[str, Any]):
        """Enhanced processing of JSON files for VASP MD with comprehensive DFT parameters."""
        temperatures = config.get('temperatures', [300.0])
        processed_count = 0
        
        # Extract KPOINTS configuration
        kpoints_config = config.pop('KPOINTS', {'type': 'gamma', 'grid': [1, 1, 1]})
        
        print(f"\nProcessing {len(json_files)} files for {len(temperatures)} temperatures each...")
        print(f"Using {config.get('GGA', 'LDA')} functional with ENCUT={config.get('ENCUT', 400)} eV")
        print(f"MD ensemble: {config.get('ensemble', 'NVT')}\n")
        
        for json_file in json_files:
            try:
                print(f"Processing {json_file.name}...", end=" ")
                
                # Extract structure using the existing processor
                atoms = extract_final_structure_from_json(json_file)
                if atoms is None:
                    print("ERROR: Could not extract structure")
                    continue
                
                # Generate input files for each temperature
                base_name = json_file.stem
                success_count = 0
                
                for temp in temperatures:
                    # Create temperature-specific directory
                    temp_dir = output_dir / base_name / f"T{int(temp)}K"
                    
                    # Update config for this temperature
                    temp_config = config.copy()
                    temp_config['TEBEG'] = temp
                    temp_config['TEEND'] = temp
                    temp_config['SYSTEM'] = f"{base_name} MD at {int(temp)}K"
                    
                    # Use the enhanced VASP MD input writer
                    if self._write_enhanced_vasp_md_input(atoms, temp_dir, temp_config, kpoints_config):
                        success_count += 1
                    else:
                        print(f"ERROR: Failed to write files for {temp}K")
                        break
                
                if success_count == len(temperatures):
                    processed_count += 1
                    print(f"OK - Generated {success_count} temperature setups")
                else:
                    print(f"PARTIAL - Generated {success_count}/{len(temperatures)} setups")
                    
            except Exception as e:
                print(f"ERROR: {str(e)}")
        
        print(f"\nSuccessfully processed {processed_count} out of {len(json_files)} JSON files.")
        print(f"Total input sets generated: {processed_count * len(temperatures)}")

    def _write_enhanced_vasp_md_input(self, atoms, output_dir: Path, config: Dict[str, Any], kpoints_config: Dict[str, Any]) -> bool:
        """Write enhanced VASP MD input files with comprehensive DFT parameters."""
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate INCAR with all the comprehensive parameters
            incar_content = self._generate_enhanced_vasp_md_incar(config)
            
            # Generate POSCAR from atoms object
            poscar_content = self._generate_poscar_from_atoms(atoms, config['SYSTEM'])
            
            # Generate KPOINTS
            kpoints_content = self._generate_vasp_kpoints(kpoints_config)
            
            # Write all files
            (output_dir / "INCAR").write_text(incar_content)
            (output_dir / "POSCAR").write_text(poscar_content)
            (output_dir / "KPOINTS").write_text(kpoints_content)
            
            # Write a run script
            run_script = self._generate_vasp_md_run_script(config)
            (output_dir / "run_vasp.sh").write_text(run_script)
            (output_dir / "run_vasp.sh").chmod(0o755)
            
            return True
            
        except Exception as e:
            print(f"Error writing VASP MD files: {e}")
            return False

    def _generate_poscar_from_atoms(self, atoms, system_name: str) -> str:
        """Generate POSCAR content from ASE atoms object."""
        lines = []
        
        # Title
        lines.append(f"{system_name}")
        lines.append("1.0")  # Scaling factor
        
        # Cell vectors
        cell = atoms.get_cell()
        for vector in cell:
            lines.append(f"  {vector[0]:15.10f}  {vector[1]:15.10f}  {vector[2]:15.10f}")
        
        # Get unique elements and counts
        symbols = atoms.get_chemical_symbols()
        from collections import Counter
        element_counts = Counter(symbols)
        
        # Element types and counts
        elements = list(element_counts.keys())
        counts = list(element_counts.values())
        
        lines.append("  " + "  ".join(elements))
        lines.append("  " + "  ".join(map(str, counts)))
        
        # Coordinate mode
        lines.append("Cartesian")
        
        # Coordinates (sorted by element type)
        positions = atoms.get_positions()
        for element in elements:
            for i, symbol in enumerate(symbols):
                if symbol == element:
                    pos = positions[i]
                    lines.append(f"  {pos[0]:15.10f}  {pos[1]:15.10f}  {pos[2]:15.10f}")
        
        return "\n".join(lines)

    def _generate_vasp_md_run_script(self, config: Dict[str, Any]) -> str:
        """Generate a run script for VASP MD calculation."""
        script_lines = [
            "#!/bin/bash",
            "# VASP MD run script",
            f"# Generated for {config.get('SYSTEM', 'VASP MD calculation')}",
            f"# Temperature: {config.get('TEBEG', 300)}K",
            f"# Ensemble: {config.get('ensemble', 'NVT')}",
            "",
            "# Set number of cores (adjust as needed)",
            "export OMP_NUM_THREADS=1",
            "",
            "# Run VASP",
            "# Uncomment and modify the appropriate line below:",
            "",
            "# For single-node run:",
            "# mpirun -np 8 vasp_std > vasp.out 2>&1",
            "",
            "# For SLURM cluster:",
            "# srun vasp_std > vasp.out 2>&1",
            "",
            "# For PBS cluster:", 
            "# mpirun vasp_std > vasp.out 2>&1",
            "",
            "echo 'VASP MD calculation completed.'",
            "echo 'Check vasp.out for output and XDATCAR for trajectory.'"
        ]
        
        return "\n".join(script_lines)

    def _handle_vasp_md_input_generation(self):
        """Direct VASP MD input generation from structure files."""
        if not check_ase_available():
            print("\nERROR: ASE is required for VASP MD input generation.")
            print("Please install it using: pip install ase\n")
            return
        
        print("\n==== VASP Molecular Dynamics Input Generation ====\n")
        
        # Get structure file path
        file_path = input("Enter the path to your structure file or directory (XYZ or CIF): ")
        input_path = Path(file_path).expanduser().resolve()
        
        if not input_path.exists():
            print(f"Error: Path '{input_path}' does not exist.")
            return
        
        # Get output directory
        output_dir = self._get_output_directory("Enter the output directory for MD input files: ")
        if not output_dir:
            return
        
        # Get MD configuration
        print("\nChoose parameter input method:")
        print("1. Template mode (predefined settings)")
        print("2. Interactive mode (custom settings)")
        
        mode = input("\nSelect mode (1/2) [2]: ").strip() or "2"
        
        if mode == "1":
            md_config = self._select_vasp_md_template()
        else:
            md_config = self._get_vasp_md_parameters_interactively()
        
        if not md_config:
            print("MD configuration was not completed. Aborting.")
            return
        
        # Process structure files
        if input_path.is_dir():
            structure_files = list(input_path.glob("*.cif")) + list(input_path.glob("*.xyz"))
            if not structure_files:
                print(f"No structure files found in {input_path}")
                return
            
            print(f"\nProcessing {len(structure_files)} structure files...")
            self._process_structure_files_vasp_md(structure_files, output_dir, md_config)
        else:
            print(f"\nProcessing single file: {input_path.name}")
            self._process_structure_files_vasp_md([input_path], output_dir, md_config)

    def _process_json_files_vasp_md(self, json_files: List[Path], output_dir: Path, config: Dict[str, Any]):
        """Process JSON files for VASP MD using the efficient processor."""
        temperatures = config.get('temperatures', [300.0])
        processed_count = 0
        
        for json_file in json_files:
            try:
                print(f"Processing {json_file.name}...", end=" ")
                
                # Extract structure using processor
                atoms = extract_final_structure_from_json(json_file)
                if atoms is None:
                    print("ERROR: Could not extract structure")
                    continue
                
                # Generate input files for each temperature
                base_name = json_file.stem
                success = True
                
                for temp in temperatures:
                    temp_dir = output_dir / base_name / f"T{int(temp)}K"
                    if not write_complete_vasp_md_input(atoms, temp_dir, temp, config):
                        success = False
                        break
                
                if success:
                    processed_count += 1
                    print(f"OK - Generated {len(temperatures)} temperature setups")
                else:
                    print("ERROR: Failed to write input files")
                    
            except Exception as e:
                print(f"ERROR: {str(e)}")
        
        print(f"\nSuccessfully processed {processed_count} out of {len(json_files)} JSON files.")

    def _process_structure_files_vasp_md(self, structure_files: List[Path], output_dir: Path, config: Dict[str, Any]):
        """Process structure files for VASP MD input generation."""
        temperatures = config.get('temperatures', [300.0])
        processed_count = 0
        
        for structure_file in structure_files:
            try:
                print(f"Processing {structure_file.name}...", end=" ")
                
                # Convert structure file to atoms
                atoms = structure_file_to_atoms(structure_file)
                if atoms is None:
                    print("ERROR: Could not read structure file")
                    continue
                
                # Generate input files for each temperature
                base_name = structure_file.stem
                success = True
                
                for temp in temperatures:
                    temp_dir = output_dir / base_name / f"T{int(temp)}K"
                    if not write_complete_vasp_md_input(atoms, temp_dir, temp, config):
                        success = False
                        break
                
                if success:
                    processed_count += 1
                    print(f"OK - Generated {len(temperatures)} temperature setups")
                else:
                    print("ERROR: Failed to write input files")
                    
            except Exception as e:
                print(f"ERROR: {str(e)}")
        
        print(f"\nSuccessfully processed {processed_count} out of {len(structure_files)} structure files.")

    def _get_thermostat_parameters(self) -> Dict[str, Any]:
        """Get thermostat-specific parameters."""
        print("\n--- Thermostat Selection ---")
        print("1. Nose-Hoover (deterministic)")
        print("2. Langevin (stochastic)")
        print("3. Andersen (stochastic)")
        print("4. CSVR (efficient)")
        print("5. Nose-Hoover Chain (improved)")
        
        choice = input("Select thermostat (1-5) [1]: ").strip() or "1"
        
        if choice == "2":
            gamma = float(input("Langevin friction (LANGEVIN_GAMMA) [10.0]: ").strip() or "10.0")
            return {"thermostat": "langevin", "langevin_gamma": gamma}
        elif choice == "3":
            prob = float(input("Andersen collision probability [0.1]: ").strip() or "0.1")
            return {"thermostat": "andersen", "andersen_prob": prob}
        elif choice == "4":
            period = int(input("CSVR period [10]: ").strip() or "10")
            return {"thermostat": "csvr", "csvr_period": period}
        elif choice == "5":
            chains = int(input("Number of chains [4]: ").strip() or "4")
            return {"thermostat": "nhc", "nhc_nchains": chains, "nhc_period": 1}
        else:
            smass = float(input("Nose mass parameter (SMASS) [0.5]: ").strip() or "0.5")
            return {"thermostat": "nose_hoover", "smass": smass}

    def _get_aimd_template(self):
        """Provide predefined templates for AIMD calculations."""
        print("\n==== AIMD Template Selection ====")
        
        print("\nAvailable AIMD templates:")
        print("1. Standard NVT (300K, 500K)")
        print("2. Multi-temperature NVT (100K, 200K, 300K, 400K, 500K)")
        print("3. High-temperature melting (1000K, 1500K, 2000K)")
        print("4. Low-temperature glass transition (50K, 100K, 150K, 200K)")
        print("5. Custom temperature set")
        
        choice = input("\nSelect template (1-5) [1]: ").strip() or "1"
        
        templates = {
            "1": {
                "name": "Standard NVT",
                "temperatures": [300.0, 500.0],
                "ensemble": "NVT",
                "timestep": 0.5,  # fs
                "steps": 1000000,
                "thermostat": "GLE",
                "print_freq": 1,   # MD steps
                "restart_freq": 500  # MD steps
            },
            "2": {
                "name": "Multi-temperature NVT",
                "temperatures": [100.0, 200.0, 300.0, 400.0, 500.0],
                "ensemble": "NVT",
                "timestep": 0.5,
                "steps": 1000000,
                "thermostat": "GLE",
                "print_freq": 1,
                "restart_freq": 500
            },
            "3": {
                "name": "High-temperature melting",
                "temperatures": [1000.0, 1500.0, 2000.0],
                "ensemble": "NVT",
                "timestep": 0.25,  # Smaller timestep for stability
                "steps": 500000,
                "thermostat": "NOSE",
                "print_freq": 1,
                "restart_freq": 250
            },
            "4": {
                "name": "Low-temperature glass transition",
                "temperatures": [50.0, 100.0, 150.0, 200.0],
                "ensemble": "NVT",
                "timestep": 1.0,   # Larger timestep for efficiency
                "steps": 2000000,
                "thermostat": "GLE",
                "print_freq": 2,    # Save less frequently
                "restart_freq": 1000
            }
        }
        
        if choice in templates:
            config = templates[choice].copy()
            
            # Show the selected template details
            print(f"\nSelected: {config['name']}")
            print(f"Temperatures: {', '.join([f'{t}K' for t in config['temperatures']])}")
            print(f"Ensemble: {config['ensemble']}")
            print(f"Timestep: {config['timestep']} fs")
            print(f"Steps: {config['steps']}")
            print(f"Thermostat: {config['thermostat']}")
            
            confirm = input("\nUse this template? (y/n) [y]: ").strip().lower() or "y"
            if confirm != "y":
                return self._get_aimd_template()
                
            return config
            
        elif choice == "5":
            # Custom temperature set
            config = templates["1"].copy()  # Start with standard template
            config["name"] = "Custom temperatures"
            
            temp_input = input("\nEnter temperatures in K (comma-separated, e.g., 200,300,400): ").strip()
            try:
                temps = [float(t.strip()) for t in temp_input.split(',') if t.strip()]
                if not temps:
                    print("No valid temperatures entered. Using default 300K.")
                    temps = [300.0]
                config["temperatures"] = temps
            except ValueError:
                print("Invalid temperature format. Using default 300K.")
                config["temperatures"] = [300.0]
            
            # Options to customize other parameters
            customize = input("\nCustomize other parameters? (y/n) [n]: ").strip().lower() or "n"
            if customize == "y":
                # Ensemble
                print("\nAvailable ensembles:")
                print("1. NVT (constant temperature)")
                print("2. NPT (constant pressure and temperature)")
                ensemble_choice = input("Select ensemble (1/2) [1]: ").strip() or "1"
                config["ensemble"] = "NPT" if ensemble_choice == "2" else "NVT"
                
                # Timestep
                timestep = input("Timestep in fs [0.5]: ").strip() or "0.5"
                try:
                    config["timestep"] = float(timestep)
                except ValueError:
                    print("Invalid timestep. Using default 0.5 fs.")
                    config["timestep"] = 0.5
                
                # Number of steps
                steps = input("Number of MD steps [1000000]: ").strip() or "1000000"
                try:
                    config["steps"] = int(steps)
                except ValueError:
                    print("Invalid steps. Using default 1000000.")
                    config["steps"] = 1000000
                
                # Thermostat
                print("\nAvailable thermostats:")
                print("1. GLE (Generalized Langevin Equation)")
                print("2. NOSE (Nosé-Hoover)")
                print("3. CSVR (Canonical Sampling through Velocity Rescaling)")
                thermostat_choice = input("Select thermostat (1/2/3) [1]: ").strip() or "1"
                if thermostat_choice == "2":
                    config["thermostat"] = "NOSE"
                elif thermostat_choice == "3":
                    config["thermostat"] = "CSVR"
                else:
                    config["thermostat"] = "GLE"
            
            return config
        else:
            print("Invalid choice. Using standard template.")
            return templates["1"]

    def _get_aimd_parameters_interactively(self):
        """Interactive questionnaire for AIMD parameters."""
        print("\n==== AIMD Calculation Settings ====")
        
        # Initialize config
        aimd_config = {
            "temperatures": [],
            "ensemble": "NVT",
            "timestep": 0.5,
            "steps": 1000000,
            "thermostat": "GLE",
            "print_freq": 1,
            "restart_freq": 500,
            "xc_functional": "PBE",
            "vdw_correction": True
        }
        
        # 1. Temperature settings
        print("\n--- Temperature Settings ---")
        temp_input = input("Enter temperatures in K (comma-separated, e.g., 200,300,400): ").strip()
        try:
            temps = [float(t.strip()) for t in temp_input.split(',') if t.strip()]
            if not temps:
                print("No valid temperatures entered. Using default 300K.")
                temps = [300.0]
            aimd_config["temperatures"] = temps
        except ValueError:
            print("Invalid temperature format. Using default 300K.")
            aimd_config["temperatures"] = [300.0]
        
        # 2. Ensemble settings
        print("\n--- Ensemble Settings ---")
        print("Available ensembles:")
        print("1. NVT (constant temperature)")
        print("2. NPT (constant pressure and temperature)")
        print("3. NVE (microcanonical)")
        
        ensemble_choice = input("Select ensemble (1/2/3) [1]: ").strip() or "1"
        if ensemble_choice == "2":
            aimd_config["ensemble"] = "NPT"
        elif ensemble_choice == "3":
            aimd_config["ensemble"] = "NVE"
        else:
            aimd_config["ensemble"] = "NVT"
        
        # 3. Thermostat settings (if not NVE)
        if aimd_config["ensemble"] != "NVE":
            print("\n--- Thermostat Settings ---")
            print("Available thermostats:")
            print("1. GLE (Generalized Langevin Equation - accurate sampling)")
            print("2. NOSE (Nosé-Hoover - deterministic)")
            print("3. CSVR (Canonical Sampling through Velocity Rescaling - efficient)")
            
            thermostat_choice = input("Select thermostat (1/2/3) [1]: ").strip() or "1"
            if thermostat_choice == "2":
                aimd_config["thermostat"] = "NOSE"
            elif thermostat_choice == "3":
                aimd_config["thermostat"] = "CSVR"
            else:
                aimd_config["thermostat"] = "GLE"
        
        # 4. MD run parameters
        print("\n--- MD Run Parameters ---")
        
        # Timestep
        timestep = input("Timestep in fs [0.5]: ").strip() or "0.5"
        try:
            aimd_config["timestep"] = float(timestep)
            if aimd_config["timestep"] <= 0:
                print("Timestep must be positive. Using default 0.5 fs.")
                aimd_config["timestep"] = 0.5
        except ValueError:
            print("Invalid timestep. Using default 0.5 fs.")
            aimd_config["timestep"] = 0.5
        
        # Number of steps
        steps = input("Number of MD steps [1000000]: ").strip() or "1000000"
        try:
            aimd_config["steps"] = int(steps)
            if aimd_config["steps"] <= 0:
                print("Number of steps must be positive. Using default 1000000.")
                aimd_config["steps"] = 1000000
        except ValueError:
            print("Invalid number of steps. Using default 1000000.")
            aimd_config["steps"] = 1000000
        
        # 5. DFT settings
        print("\n--- DFT Settings ---")
        
        # Exchange-correlation functional
        print("Available XC functionals:")
        print("1. PBE (general purpose)")
        print("2. revPBE (revised PBE, better for some systems)")
        print("3. BLYP (better for some molecular systems)")
        
        xc_choice = input("Select XC functional (1/2/3) [1]: ").strip() or "1"
        if xc_choice == "2":
            aimd_config["xc_functional"] = "revPBE"
        elif xc_choice == "3":
            aimd_config["xc_functional"] = "BLYP"
        else:
            aimd_config["xc_functional"] = "PBE"
        
        # vdW correction
        vdw = input("Include van der Waals correction? (y/n) [y]: ").strip().lower() or "y"
        aimd_config["vdw_correction"] = (vdw == "y")
        
        # 6. Output settings
        print("\n--- Output Settings ---")
        
        # Trajectory printing frequency
        print_freq = input("Save trajectory every N steps [1]: ").strip() or "1"
        try:
            aimd_config["print_freq"] = int(print_freq)
            if aimd_config["print_freq"] <= 0:
                print("Print frequency must be positive. Using default 1.")
                aimd_config["print_freq"] = 1
        except ValueError:
            print("Invalid print frequency. Using default 1.")
            aimd_config["print_freq"] = 1
        
        # Restart file frequency
        restart_freq = input("Save restart files every N steps [500]: ").strip() or "500"
        try:
            aimd_config["restart_freq"] = int(restart_freq)
            if aimd_config["restart_freq"] <= 0:
                print("Restart frequency must be positive. Using default 500.")
                aimd_config["restart_freq"] = 500
        except ValueError:
            print("Invalid restart frequency. Using default 500.")
            aimd_config["restart_freq"] = 500
        
        # 7. Summary
        print("\n=== Summary of AIMD Settings ===")
        print(f"Temperatures: {', '.join([f'{t}K' for t in aimd_config['temperatures']])}")
        print(f"Ensemble: {aimd_config['ensemble']}")
        if aimd_config["ensemble"] != "NVE":
            print(f"Thermostat: {aimd_config['thermostat']}")
        print(f"Timestep: {aimd_config['timestep']} fs")
        print(f"Steps: {aimd_config['steps']}")
        print(f"XC Functional: {aimd_config['xc_functional']}")
        print(f"vdW Correction: {'Yes' if aimd_config['vdw_correction'] else 'No'}")
        print(f"Save trajectory every: {aimd_config['print_freq']} steps")
        print(f"Save restart files every: {aimd_config['restart_freq']} steps")
        
        confirm = input("\nConfirm these settings? (y/n) [y]: ").strip().lower() or "y"
        if confirm != "y":
            print("Restarting parameter selection...")
            return self._get_aimd_parameters_interactively()
        
        return aimd_config

    def _write_xyz_for_aimd(self, atoms, output_path):
        """Write an XYZ file from the given ASE Atoms object."""
        symbols = atoms.get_chemical_symbols()
        positions = atoms.get_positions()
        
        with open(output_path, 'w') as f:
            f.write(f"{len(atoms)}\n")
            f.write("XYZ from AIMD processing\n")
            for sym, pos in zip(symbols, positions):
                f.write(f"{sym} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}\n")

    def _write_cp2k_aimd_input(self, atoms, output_path, temperature, config):
        """
        Write a CP2K AIMD input file based on the provided configuration.
        
        Args:
            atoms: ASE Atoms object
            output_path: Path to output CP2K input file
            temperature: MD simulation temperature
            config: Configuration dictionary with AIMD settings
        """
        # Extract cell parameters
        cell_vectors = atoms.cell.array
        cellpars = cell_to_cellpar(cell_vectors)  # [a, b, c, alpha, beta, gamma]
        ABC = cellpars[:3]
        alpha_beta_gamma = cellpars[3:]
        
        # Extract atomic symbols and positions
        atom_symbols = atoms.get_chemical_symbols()
        atom_positions = atoms.get_positions()
        
        # Standard mapping for basis sets and potentials
        kind_parameters = {
            "H":  {"BASIS_SET": "ORB TZV2P-MOLOPT-PBE-GTH-q1", "POTENTIAL": "GTH-PBE-q1"},
            "C":  {"BASIS_SET": "ORB TZV2P-MOLOPT-PBE-GTH-q4", "POTENTIAL": "GTH-PBE-q4"},
            "N":  {"BASIS_SET": "ORB TZV2P-MOLOPT-PBE-GTH-q5", "POTENTIAL": "GTH-PBE-q5"},
            "O":  {"BASIS_SET": "ORB TZV2P-MOLOPT-PBE-GTH-q6", "POTENTIAL": "GTH-PBE-q6"},
            "F":  {"BASIS_SET": "ORB aug-TZV2P-GTH-q7",         "POTENTIAL": "GTH-PBE-q7"},
            "Si": {"BASIS_SET": "ORB aug-TZV2P-GTH-q4",         "POTENTIAL": "GTH-PBE-q4"},
            "Cl": {"BASIS_SET": "ORB aug-TZV2P-GTH-q7",         "POTENTIAL": "GTH-PBE-q7"},
            "Br": {"BASIS_SET": "ORB TZV2P-MOLOPT-PBE-GTH-q7",  "POTENTIAL": "GTH-PBE-q7"},
            "I":  {"BASIS_SET": "ORB TZV2P-MOLOPT-PBE-GTH-q7",  "POTENTIAL": "GTH-PBE-q7"}
        }
        
        # Create CP2K input file content
        lines = []
        
        # Global section
        lines.append("&GLOBAL")
        lines.append(f"  PROJECT AIMD_T{temperature}")
        lines.append("  RUN_TYPE MD")
        lines.append("  PRINT_LEVEL LOW")
        lines.append("&END GLOBAL\n")
        
        # Force evaluation section
        lines.append("&FORCE_EVAL")
        lines.append("  METHOD Quickstep")
        lines.append("  &DFT")
        lines.append("    BASIS_SET_FILE_NAME BASIS_MOLOPT")
        lines.append("    POTENTIAL_FILE_NAME GTH_POTENTIALS")
        
        # MGRID section
        lines.append("    &MGRID")
        lines.append("      CUTOFF 500")
        lines.append("      REL_CUTOFF 50")
        lines.append("      NGRIDS 4")
        lines.append("    &END MGRID")
        
        # QS section
        lines.append("    &QS")
        lines.append("      EPS_DEFAULT 1.0E-12")
        lines.append("      EXTRAPOLATION ASPC")
        lines.append("    &END QS")
        
        # SCF section
        lines.append("    &SCF")
        lines.append("      SCF_GUESS ATOMIC")
        lines.append("      MAX_SCF 30")
        lines.append("      EPS_SCF 1.0E-6")
        lines.append("      &OT")
        lines.append("        MINIMIZER DIIS")
        lines.append("        PRECONDITIONER FULL_SINGLE_INVERSE")
        lines.append("      &END OT")
        lines.append("      &OUTER_SCF")
        lines.append("        MAX_SCF 20")
        lines.append("        EPS_SCF 1.0E-6")
        lines.append("      &END OUTER_SCF")
        lines.append("      &PRINT")
        lines.append("        &RESTART")
        lines.append("          ADD_LAST NUMERIC")
        lines.append("          &EACH")
        lines.append("            QS_SCF 0")
        lines.append("          &END EACH")
        lines.append("        &END RESTART")
        lines.append("      &END PRINT")
        lines.append("    &END SCF")
        
        # XC section with functional and vdW correction if enabled
        xc_func = config.get("xc_functional", "PBE")
        vdw_correction = config.get("vdw_correction", True)
        
        lines.append("    &XC")
        lines.append("      &XC_FUNCTIONAL")
        
        if xc_func == "PBE":
            lines.append("        &PBE")
            lines.append("        &END PBE")
        elif xc_func == "revPBE":
            lines.append("        &PBE")
            lines.append("          PARAMETRIZATION REVPBE")
            lines.append("        &END PBE")
        elif xc_func == "BLYP":
            lines.append("        &BLYP")
            lines.append("        &END BLYP")
        else:
            # Default to PBE
            lines.append("        &PBE")
            lines.append("        &END PBE")
        
        lines.append("      &END XC_FUNCTIONAL")
        
        if vdw_correction:
            lines.append("      &VDW_POTENTIAL")
            lines.append("        POTENTIAL_TYPE PAIR_POTENTIAL")
            lines.append("        &PAIR_POTENTIAL")
            lines.append("          TYPE DFTD3")
            lines.append("          R_CUTOFF 12.0")
            lines.append("          LONG_RANGE_CORRECTION TRUE")
            
            # Reference functional for D3 correction
            ref_func = "PBE"
            if xc_func == "revPBE":
                ref_func = "revPBE"
            elif xc_func == "BLYP":
                ref_func = "BLYP"
            
            lines.append(f"          REFERENCE_FUNCTIONAL {ref_func}")
            lines.append("          PARAMETER_FILE_NAME dftd3.dat")
            lines.append("        &END PAIR_POTENTIAL")
            lines.append("      &END VDW_POTENTIAL")
        
        lines.append("    &END XC")
        lines.append("  &END DFT")
        
        # Print forces
        lines.append("  &PRINT")
        lines.append("    &FORCES ON")
        lines.append("    &END FORCES")
        lines.append("  &END PRINT")
        
        # Subsystem section (cell and coordinates)
        lines.append("  &SUBSYS")
        lines.append("    &CELL")
        lines.append(f"      ABC [angstrom] {ABC[0]:.6f} {ABC[1]:.6f} {ABC[2]:.6f}")
        lines.append(f"      ALPHA_BETA_GAMMA [deg] {alpha_beta_gamma[0]:.6f} {alpha_beta_gamma[1]:.6f} {alpha_beta_gamma[2]:.6f}")
        lines.append("      PERIODIC XYZ")
        lines.append("    &END CELL")
        
        # Add KIND sections for each element type
        for atom in set(atom_symbols):
            if atom in kind_parameters:
                params = kind_parameters[atom]
                lines.append(f"    &KIND {atom}")
                lines.append(f"      BASIS_SET {params['BASIS_SET']}")
                lines.append(f"      POTENTIAL {params['POTENTIAL']}")
                lines.append("    &END KIND")
            else:
                lines.append(f"    &KIND {atom}")
                lines.append("      BASIS_SET DZVP-MOLOPT-SR-GTH")
                lines.append("      POTENTIAL GTH-PBE-q0")
                lines.append("    &END KIND")
        
        # Add atomic coordinates
        lines.append("    &COORD")
        for sym, pos in zip(atom_symbols, atom_positions):
            lines.append(f"      {sym} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}")
        lines.append("    &END COORD")
        lines.append("  &END SUBSYS")
        lines.append("&END FORCE_EVAL\n")
        
        # MOTION section with MD settings
        lines.append("&MOTION")
        
        # MD section
        ensemble = config.get("ensemble", "NVT")
        timestep = config.get("timestep", 0.5)
        steps = config.get("steps", 1000000)
        
        lines.append("  &MD")
        lines.append(f"    ENSEMBLE {ensemble}")
        lines.append(f"    TEMPERATURE [K] {temperature}")
        lines.append(f"    TIMESTEP [fs] {timestep}")
        lines.append(f"    STEPS {steps}")
        
        # Thermostat section (if not NVE)
        if ensemble != "NVE":
            thermostat = config.get("thermostat", "GLE")
            lines.append("    &THERMOSTAT")
            lines.append("      REGION MASSIVE")
            lines.append(f"      TYPE {thermostat}")
            
            if thermostat == "GLE":
                lines.append("      &GLE")
                lines.append("        NDIM 5")
                lines.append("        A_SCALE [ps^-1] 1.00")
                lines.append("        A_LIST    1.859575861256e+2   2.726385349840e-1   1.152610045461e+1  -3.641457826260e+1   2.317337581602e+2")
                lines.append("        A_LIST   -2.780952471206e-1   8.595159180871e-5   7.218904801765e-1  -1.984453934386e-1   4.240925758342e-1")
                lines.append("        A_LIST   -1.482580813121e+1  -7.218904801765e-1   1.359090212128e+0   5.149889628035e+0  -9.994926845099e+0")
                lines.append("        A_LIST   -1.037218912688e+1   1.984453934386e-1  -5.149889628035e+0   2.666191089117e+1   1.150771549531e+1")
                lines.append("        A_LIST    2.180134636042e+2  -4.240925758342e-1   9.994926845099e+0  -1.150771549531e+1   3.095839456559e+2")
                lines.append("      &END GLE")
            elif thermostat == "NOSE":
                lines.append("      &NOSE")
                lines.append("        LENGTH 3")
                lines.append("        YOSHIDA 3")
                lines.append("        TIMECON [fs] 100.0")
                lines.append("        MTS 2")
                lines.append("      &END NOSE")
            elif thermostat == "CSVR":
                lines.append("      &CSVR")
                lines.append("        TIMECON [fs] 100.0")
                lines.append("      &END CSVR")
            
            lines.append("    &END THERMOSTAT")
        
        # Barostat for NPT
        if ensemble == "NPT":
            lines.append("    &BAROSTAT")
            lines.append("      PRESSURE [bar] 1.0")
            lines.append("      TIMECON [fs] 100.0")
            lines.append("    &END BAROSTAT")
        
        lines.append("  &END MD")
        
        # Print settings for trajectory, forces, etc.
        print_freq = config.get("print_freq", 1)
        restart_freq = config.get("restart_freq", 500)
        
        lines.append("  &PRINT")
        lines.append("    &TRAJECTORY")
        lines.append("      FORMAT XYZ")
        lines.append("      UNIT angstrom")
        lines.append("      &EACH")
        lines.append(f"        MD {print_freq}")
        lines.append("      &END EACH")
        lines.append("    &END TRAJECTORY")
        lines.append("    &VELOCITIES OFF")
        lines.append("    &END VELOCITIES")
        lines.append("    &FORCES ON")
        lines.append("    &END FORCES")
        lines.append("    &RESTART_HISTORY")
        lines.append("      &EACH")
        lines.append(f"        MD {restart_freq}")
        lines.append("      &END EACH")
        lines.append("    &END RESTART_HISTORY")
        lines.append("  &END PRINT")
        lines.append("&END MOTION")
        
        # Write the content to the file
        with open(output_path, 'w') as f:
            f.write("\n".join(lines))

    def _get_aimd_config(self):
        """Get AIMD processing configuration from user input or config file."""
        print("AIMD configuration can be provided via a YAML file or directly through this interface.")
        use_file = input("Use a config file? (y/n) [n]: ").strip().lower() or "n"
        
        if use_file == "y":
            config_path = input("Path to YAML configuration file: ").strip()
            try:
                with open(config_path, "r") as f:
                    config = yaml.safe_load(f)
                
                # Validate config content
                required_keys = ['json_dir', 'output_dir', 'temperatures']
                missing_keys = [key for key in required_keys if key not in config]
                
                if missing_keys:
                    print(f"Error: Missing required configuration keys: {', '.join(missing_keys)}")
                    print("Please make sure your configuration file includes these keys.")
                    return None
                
                return config
            except Exception as e:
                print(f"Error loading configuration file: {e}")
                return None
        else:
            # Get configuration via interactive prompts
            config = {}
            
            # Get JSON directory
            json_dir = input("Directory containing JSON files: ").strip()
            if not json_dir:
                print("Error: JSON directory is required.")
                return None
            
            json_dir_path = Path(json_dir).expanduser().resolve()
            if not json_dir_path.exists() or not json_dir_path.is_dir():
                print(f"Error: Directory not found or not a directory: {json_dir}")
                return None
            
            config['json_dir'] = str(json_dir_path)
            
            # Get output directory
            output_dir = input("Output directory for generated files: ").strip()
            if not output_dir:
                print("Error: Output directory is required.")
                return None
            
            output_dir_path = Path(output_dir).expanduser().resolve()
            output_dir_path.mkdir(parents=True, exist_ok=True)
            config['output_dir'] = str(output_dir_path)
            
            # Get temperatures
            temps_input = input("list of MD temperatures (comma-separated, e.g., 200,250,300): ").strip()
            if not temps_input:
                print("Error: At least one temperature is required.")
                return None
            
            try:
                temperatures = [float(t.strip()) for t in temps_input.split(',')]
                if not temperatures:
                    raise ValueError("No valid temperatures provided")
                config['temperatures'] = temperatures
            except ValueError as e:
                print(f"Error parsing temperatures: {e}")
                return None
            
            # Optional melting point
            melting_point = input("Melting point in K (optional, press Enter to skip): ").strip()
            if melting_point:
                try:
                    config['melting_point'] = float(melting_point)
                except ValueError:
                    print("Warning: Invalid melting point value. Skipping.")
            
            return config

    def _process_aimd_json_files(self, json_dir, temperatures, output_dir):
        """
        Process each JSON file in the specified directory for AIMD calculations.
        
        For each JSON file:
          - Load the JSON data (expected to be a list of configurations).
          - Extract the last configuration as the final geometry.
          - Reconstruct an ASE Atoms object.
          - Write a new XYZ file and CP2K MD input files (one per temperature).
        """
        json_pattern = os.path.join(json_dir, "*.json")
        json_files = glob.glob(json_pattern)
        
        if not json_files:
            print(f"[ERROR] No JSON files found in {json_dir}")
            return
        
        print(f"\nFound {len(json_files)} JSON files in {json_dir}")
        print(f"Will generate CP2K input files for {len(temperatures)} temperatures: {temperatures}")
        print(f"Output will be saved in {output_dir}\n")
        
        processed_count = 0
        for filepath in json_files:
            try:
                file_path = Path(filepath)
                base_name = file_path.stem
                
                print(f"Processing: {file_path.name}...", end=" ")
                
                # Load JSON data and extract the last configuration
                with file_path.open('r') as f:
                    data = json.load(f)
                
                if not data:
                    print("WARNING: Empty JSON file")
                    continue
                    
                final_config = data[-1]
                
                # Reconstruct ASE Atoms object
                coords = final_config.get("coordinates", [])
                if not coords:
                    print("WARNING: No coordinates found")
                    continue
                    
                positions = np.array([[c["x"], c["y"], c["z"]] for c in coords], dtype=np.float32)
                symbols = final_config.get("atom_types", [])
                cell_lengths = final_config.get("cell_lengths", [])
                cell_angles = final_config.get("cell_angles", [])
                
                if not cell_lengths or not cell_angles:
                    print("WARNING: Missing cell parameters")
                    continue
                    
                cellpars = cell_lengths + cell_angles
                cell_vectors = cellpar_to_cell(cellpars)
                atoms_obj = Atoms(symbols=symbols, positions=positions, cell=cell_vectors, pbc=True)
                
                # Define output directories within the specified output_dir
                base_out = Path(output_dir) / base_name
                xyz_out_dir = base_out / "XYZ"
                cp2k_out_dir = base_out / "CP2K"
                xyz_out_dir.mkdir(parents=True, exist_ok=True)
                cp2k_out_dir.mkdir(parents=True, exist_ok=True)
                
                # Write new XYZ file
                xyz_path = xyz_out_dir / f"{base_name}_original.xyz"
                self._write_xyz_for_aimd(atoms_obj, xyz_path)
                
                # Write CP2K input files for each specified temperature
                for temp in temperatures:
                    cp2k_path = cp2k_out_dir / f"T{temp}_{base_name}_cp2k_input.inp"
                    self._write_cp2k_aimd_input(atoms_obj, cp2k_path, temperature=temp)
                
                processed_count += 1
                print("OK")
                
            except Exception as exc:
                print(f"ERROR: {exc}")
        
        print(f"\nProcessed {processed_count} out of {len(json_files)} JSON files.")
        print(f"Generated {processed_count * len(temperatures)} CP2K AIMD input files.")

    def _process_json_files_cp2k_md(self, json_files: List[Path], output_dir: Path, aimd_config: Dict[str, Any]):
        """Process JSON files for CP2K MD."""
        temperatures = aimd_config.get('temperatures', [300.0])
        processed_count = 0
        
        for json_file in json_files:
            try:
                print(f"Processing {json_file.name}...", end=" ")
                
                # Extract final configuration
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                if not data:
                    print("ERROR: Empty JSON file")
                    continue
                    
                final_config = data[-1]
                
                # Check required fields
                if not all(k in final_config for k in ["coordinates", "atom_types", "cell_lengths", "cell_angles"]):
                    print("ERROR: Missing required fields in JSON")
                    continue
                
                # Create ASE Atoms object
                coords = final_config.get("coordinates", [])
                positions = np.array([[c["x"], c["y"], c["z"]] for c in coords], dtype=np.float32)
                symbols = final_config.get("atom_types", [])
                cell_lengths = final_config.get("cell_lengths", [])
                cell_angles = final_config.get("cell_angles", [])
                
                cellpars = cell_lengths + cell_angles
                cell_vectors = cellpar_to_cell(cellpars)
                atoms_obj = Atoms(symbols=symbols, positions=positions, cell=cell_vectors, pbc=True)
                
                # Create output directories
                base_name = json_file.stem
                struct_dir = output_dir / base_name
                xyz_dir = struct_dir / "xyz"
                cp2k_dir = struct_dir / "cp2k"
                xyz_dir.mkdir(parents=True, exist_ok=True)
                cp2k_dir.mkdir(parents=True, exist_ok=True)
                
                # Write XYZ file
                xyz_path = xyz_dir / f"{base_name}.xyz"
                self._write_xyz_for_aimd(atoms_obj, xyz_path)
                
                # Write CP2K input files for each temperature
                temps = aimd_config.get('temperatures', [300.0])
                for temp in temps:
                    cp2k_path = cp2k_dir / f"{base_name}_T{int(temp)}K.inp"
                    self._write_cp2k_aimd_input(atoms_obj, cp2k_path, temp, aimd_config)
                
                processed_count += 1
                print(f"OK - Generated {len(temps)} CP2K input files")
                
            except Exception as e:
                print(f"ERROR: {str(e)}")
        
        print(f"\nSuccessfully processed {processed_count} out of {len(json_files)} JSON files.")

    # ============================================================================
    # UTILITY METHODS
    # ============================================================================

    def _get_json_input_path(self) -> Optional[Path]:
        """Get and validate JSON input path."""
        json_path = input("Enter path to JSON file or directory: ").strip()
        if not json_path:
            print("Error: No path provided.")
            return None
        
        path = Path(json_path).expanduser().resolve()
        if not path.exists():
            print(f"Error: Path does not exist: {path}")
            return None
        
        if path.is_file() and not path.suffix.lower() == '.json':
            print(f"Error: File is not a JSON file: {path}")
            return None
        
        return path

    def _get_output_directory(self, prompt: str) -> Optional[Path]:
        """Get and create output directory."""
        output_path = input(prompt).strip()
        if not output_path:
            print("Error: No output directory provided.")
            return None
        
        output_dir = Path(output_path).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

def main():
    # Create and run the system
    system = MultiAgentSystem()
    system.run()

if __name__ == "__main__":
    # Run the batch fix script if requested
    if len(sys.argv) > 1 and sys.argv[1] == "--fix-cif":
        from fix_cif_files import fix_cif_files
        fix_cif_files(".", create_backup=True)
        sys.exit(0)
    
    # Otherwise, run the main system
    main()
