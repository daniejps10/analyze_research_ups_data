import pandas as pd
from src.utils import util
from src import charts as ch
from src.data import loader
from src.config import settings

def analyze_projects_data(year: int):
   #Analyze projects data
   #####################################################################################
   proy_data = []
   proy_data.append(settings.LINE)
   #Get all project data
   proy_df = loader.get_proy_data()
   #Capitalize values
   proy_df['GRUPO_SEDE'] = proy_df['GRUPO_SEDE'].str.capitalize()
   proy_df['ODS'] = proy_df['ODS'].str.capitalize()
   proy_df['PROGRAMA_VINCULACION'] = proy_df['PROGRAMA_VINCULACION'].str.capitalize()
   #Rename columns
   proy_df = proy_df.rename(columns={
      'GRUPO_SEDE': 'Sede',
      'ODS': 'Objetivos de Desarrollo Sostenible (ODS)',
      'PROGRAMA_VINCULACION': 'Programa de Vinculación'
   })

   #Filter by RANGO_INICIO equals to year
   mask = proy_df['RANGO_INICIO'].isin([year])
   current_proy_df = proy_df[mask]
   total_current_proy = current_proy_df['CODIGO_PROYECTO'].nunique()
   proy_data.append(f"Total de proyectos en el año {year}: {total_current_proy}")

   #Filter by RANGO_INICIO equals to previous year
   mask = proy_df['RANGO_INICIO'].isin([year-1])
   previous_proy_df = proy_df[mask]
   total_previous_proy = previous_proy_df['CODIGO_PROYECTO'].nunique()
   proy_data.append(f"Total de proyectos en el año {year-1}: {total_previous_proy}")
   #Calculate percentage growth
   proy_growth = util.calculate_percentage_growth(total_previous_proy, total_current_proy)
   proy_data.append(f"Crecimiento porcentual de proyectos de {year-1} a {year}: {proy_growth:.2f}%")

   #Count number of projects by Sede
   proy_sede_df = util.count_unique_data_by_column(current_proy_df,
                                                ['Sede'],
                                                'CODIGO_PROYECTO',
                                                'N.Proyectos')
   total_proy_sede = proy_sede_df['N.Proyectos'].sum()
   proy_data.append(f"Total de proyectos por sede en el año {year}: Total -> {total_proy_sede}")
   proy_data.append(proy_sede_df.to_string(index=False))
   ch.plot_pie_chart(proy_sede_df,
                     category_col='Sede',
                     value_col='N.Proyectos')
   
   #Count number of projects by ODS
   proy_ods_df = current_proy_df[current_proy_df['Objetivos de Desarrollo Sostenible (ODS)'].notna()]
   total_proy_ods = proy_ods_df['CODIGO_PROYECTO'].nunique()
   proy_data.append(f"Total de proyectos asociados a ODS en el año {year}: {total_proy_ods}")
   proy_ods_df = util.count_unique_data_by_column(current_proy_df,
                                                ['Objetivos de Desarrollo Sostenible (ODS)'],
                                                'CODIGO_PROYECTO',
                                                'N.Proyectos')
   proy_data.append(proy_ods_df.to_string(index=False))
   ch.plot_horizontal_bar_chart(proy_ods_df,
                              category_col='Objetivos de Desarrollo Sostenible (ODS)',
                              value_col='N.Proyectos')
   
   #Count number of projects by PROGRAMA_VINCULACION
   proy_prog_df = current_proy_df[current_proy_df['Programa de Vinculación'].notna()]
   total_proy_prog = proy_prog_df['CODIGO_PROYECTO'].nunique()
   proy_data.append(f"Total de proyectos asociados a programa de vinculación en el año {year}: {total_proy_prog}")
   proy_prog_df = util.count_unique_data_by_column(current_proy_df,
                                                ['Programa de Vinculación'],
                                                'CODIGO_PROYECTO',
                                                'N.Proyectos')
   proy_data.append(proy_prog_df.to_string(index=False))
   ch.plot_horizontal_bar_chart(proy_prog_df,
                              category_col='Programa de Vinculación',
                              value_col='N.Proyectos')
   
   #Print projects data
   print('\n'.join(proy_data))
   #Save data to txt file
   util.save_txt_file('\n'.join(proy_data), f'output/projects_data_{year}.txt')
