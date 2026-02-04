import warnings
from src.analysis import groups, areas, publications, projects, rector_report, institutions

# Suppress openpyxl warnings
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

if __name__ == '__main__':
   #Step 1: Analyze groups data
   #groups.analyze_groups_data(year=2025)
   
   #Step 2: Analyze areas data
   #areas.analyze_areas_data(year=2025)
   
   #Step 3: Analyze publications
   #publications.analyze_publications_data(year=2025)
   
   #Step 4: Analyze projects
   #projects.analyze_projects_data(year=2025)
   
   #Step 5: Generate graphs for rector's book
   rector_report.generate_rector_book_graphs()
   
   #Step 6: Process institutions data
   #institutions.process_institutions_data(year=2025)