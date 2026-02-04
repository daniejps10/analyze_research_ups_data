import pandas as pd
from src.utils import util
from src import charts as ch
from src.data import loader
from src.config import settings

def analyze_groups_data(year: int = 2025):
   print(settings.LINE)
   print("Analyzing groups data...")
   print(settings.LINE)

   groups_data = []
   print("Getting previous year groups data...")
   #Read researchers and projects data
   inv_groups_df = loader.get_affiliations_data(year=year-1)
   
   #Count researchers by Genero
   inv_sede_type_title_prev_df = util.count_unique_data_by_column(inv_groups_df, 
                                                         ['Género'], 
                                                         'IDENTIFICADOR', 'N.Investigadores')
   total_inv_prev = inv_sede_type_title_prev_df['N.Investigadores'].sum()
   groups_data.append(f"Total investigadores {year-1}: {total_inv_prev}")
   groups_data.append(inv_sede_type_title_prev_df.to_string(index=False))

   groups_data.append(settings.LINE)
   print("Analyzing groups data...")
   #Read researchers and projects data
   inv_groups_df = loader.get_affiliations_data(year=year)
   
   #Count researchers by Genero
   inv_sede_type_title_df = util.count_unique_data_by_column(inv_groups_df, 
                                                         ['Género'], 
                                                         'IDENTIFICADOR', 'N.Investigadores')
   total_inv = inv_sede_type_title_df['N.Investigadores'].sum()
   groups_data.append(f"Total investigadores {year}: {total_inv}")
   groups_data.append(inv_sede_type_title_df.to_string(index=False))

   #Calculate percentage growth
   inv_growth = util.calculate_percentage_growth(total_inv_prev, total_inv)
   groups_data.append(f"Crecimiento porcentual de investigadores {year-1} a {year}: {inv_growth:.2f}%")

   inv_growth_df = pd.merge(inv_sede_type_title_prev_df, inv_sede_type_title_df, 
                           on='Género', suffixes=(f'_{year-1}', f'_{year}'))
   inv_growth_df = util.calculate_percentage_growth_df(inv_growth_df, 
                                                      f'N.Investigadores_{year-1}', 
                                                      f'N.Investigadores_{year}')
   groups_data.append(f"Crecimiento porcentual de investigadores por género {year-1} a {year}:")
   groups_data.append(inv_growth_df.to_string(index=False))

   #Count researchers by Rol en la UPS
   inv_gen_rol_df = util.count_unique_data_by_column(inv_groups_df, 
                                                   ['Género','Rol en la UPS'], 
                                                   'IDENTIFICADOR', 'N.Investigadores')
   total_inv = inv_gen_rol_df['N.Investigadores'].sum()
   groups_data.append(f"Investigadores por rol {year}: Total -> {total_inv}")
   groups_data.append(inv_gen_rol_df.to_string(index=False))
   ch.plot_distribution_by_group(inv_gen_rol_df, 
                                 group_col="Género", 
                                 category_col="Rol en la UPS", 
                                 value_col="N.Investigadores", 
                                 kind="pie",
                                 threshold_pct=4,
                                 grouped_others_min_count=2,
                                 fig_size_x=8)
   
   

   #Count by Gender and Title
   inv_gender_title_df = util.count_unique_data_by_column(inv_groups_df, 
                                                         ['Género','Titulo'], 
                                                         'IDENTIFICADOR', 'N.Investigadores')
   total_inv = inv_gender_title_df['N.Investigadores'].sum()
   groups_data.append(f"Investigadores por título académico y género {year}: Total -> {total_inv}")
   groups_data.append(inv_gender_title_df.to_string(index=False))
   ch.plot_distribution_by_group(inv_gender_title_df, 
                                 group_col="Género", 
                                 category_col="Titulo", 
                                 value_col="N.Investigadores", 
                                 kind="bar")

   #Count by Sede, Title and Genero
   inv_sede_title_gender_df = util.count_unique_data_by_column(inv_groups_df, 
                                                         ['Sede','Titulo','Género'], 
                                                         'IDENTIFICADOR', 'N.Investigadores')
   total_inv = inv_sede_title_gender_df['N.Investigadores'].sum()
   groups_data.append(f"Investigadores por sede, género y título académico {year}: Total -> {total_inv}")
   groups_data.append(inv_sede_title_gender_df.to_string(index=False))
   ch.plot_distribution_by_two_groups(inv_sede_title_gender_df, 
                                    group_col="Sede", 
                                    subgroup_col="Titulo", 
                                    category_col="Género", 
                                    value_col="N.Investigadores", 
                                    kind="pie")
   #Count by Sede, Género and Title
   inv_sede_gender_title_df = util.count_unique_data_by_column(inv_groups_df, 
                                                         ['Sede','Género','Titulo'], 
                                                         'IDENTIFICADOR', 'N.Investigadores')
   total_inv = inv_sede_gender_title_df['N.Investigadores'].sum()
   groups_data.append(f"Investigadores por sede, título académico y género {year}: Total -> {total_inv}")
   groups_data.append(inv_sede_gender_title_df.to_string(index=False))
   ch.plot_distribution_by_two_groups(inv_sede_gender_title_df, 
                                    group_col="Sede", 
                                    subgroup_col="Género", 
                                    category_col="Titulo", 
                                    value_col="N.Investigadores", 
                                    kind="bar")

   #Print groups data
   print('\n'.join(groups_data))
   #Save data to txt file
   util.save_txt_file('\n'.join(groups_data), f'output/groups_data_{year}.txt')
