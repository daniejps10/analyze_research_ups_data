from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from adjustText import adjust_text  # Import adjustText
import charts as ch

LINE = '-'*90
SGI_STATS_XLSX = 'input/sgi_stats.xlsx'
CITES_SCOPUS_XLSX = 'input/Publications_Cites_Scopus.xlsx'
INV_SCOPUS_H_INDEX_XLSX = 'input/Researcher_Scopus_H_Index.xlsx'
INV_CITATIONS_SCOPUS_XLSX = 'input/Publications_Cites_Scopus.xlsx'
COLABORATORS_XLSX = 'input/colaboradores_07_10_2025.xlsx'
INV_PROFILES_XLSX = 'input/inv_profiles.xlsx'
ALL_PURE_PUBLICATIONS_XLSX = 'input/All_Pure_Publications.xlsx'
ALL_SJR_JCR_PUBLICATIONS_XLSX = 'input/all_publications_sjr_jcr.xlsx'
SCOPUS_PUB_XLSX = 'input/scopus.xlsx'
WOS_PUB_XLSX = 'input/savedrecs.xls'

##############################################################################
# Helpers functions
##############################################################################

def save_dataframes_to_excel(dataframes_list: list, file_path: str):
   # Create a Pandas Excel writer using XlsxWriter as the engine
   with pd.ExcelWriter(file_path, engine='xlsxwriter') as writer:
      # Loop through the list and write each DataFrame to a separate sheet
      for sheet_name, df in dataframes_list:
         df.to_excel(writer, sheet_name=sheet_name, index=False)
   print(f"Excel file saved successfully at {file_path}")

def save_txt_file(content: str, file_path: str):
   with open(file_path, 'w') as file:
      file.write(content)
   print(f"Text file saved successfully at {file_path}")

def __calculate_percentage_growth(initial_value: float, end_value: float) -> float:
   #Apply formula (Vf-Vi)/Vi * 100
   if initial_value == 0:
      return 0.0
   return ((end_value - initial_value) / initial_value) * 100

def __calculate_percentage_growth_df(df: pd.DataFrame,
                                    col_init_values: str,
                                    col_end_values: str) -> pd.DataFrame:
   #Apply formula (Vf-Vi)/Vi * 100
   df["% Crecimiento"] = df.apply(
      lambda row: __calculate_percentage_growth(row[col_init_values], row[col_end_values]), axis=1
   )
   return df.drop_duplicates()

def __count_unique_data_by_column(df: pd.DataFrame, 
                                 category_cols: list[str],
                                 count_col: str,
                                 new_name_col: str) -> pd.DataFrame:
   """Count unique values in `count_col` grouped by `category_cols`
   and calculate the percentage within each category group.
   """

   # Count unique values
   count_df = (
      df.groupby(category_cols)
         .agg({count_col: "nunique"})
         .reset_index()
         .rename(columns={count_col: new_name_col})
   )

   # Total counts per category (for denominator)
   totals = (
      count_df.groupby(category_cols[:-1])[new_name_col].transform("sum")
      if len(category_cols) > 1 else count_df[new_name_col].sum()
   )

   # Calculate percentage relative to category
   count_df["Porcentaje"] = (count_df[new_name_col] / totals) * 100

   return count_df

def __calcula_anual_percentage_growth_df(df: pd.DataFrame,
                                          init_year: int, 
                                          end_year: int,
                                          init_year_col: str = None,
                                          end_year_col: str = None) -> pd.DataFrame:
      """
      Calculates the Compound Annual Growth Rate (CAGR) between two years.
      """
      # 1. Standardize column names
      col_start = init_year if init_year_col is None else init_year_col
      col_end = end_year if end_year_col is None else end_year_col
      
      # 2. Calculate the period (n)
      # Using (end - start) is standard for annual compounding
      n_periods = end_year - init_year
      
      if n_periods <= 0:
         raise ValueError("end_year must be greater than init_year")

      # 3. Safe Calculation
      # We use numpy.where to handle division by zero or negative values if they exist
      start_vals = df[col_start]
      end_vals = df[col_end]
      
      # CAGR Formula: [(End / Start)^(1/n)] - 1
      growth_series = (end_vals / start_vals) ** (1 / n_periods) - 1
      
      # 4. Clean up results (handle Inf or NaN from division by zero)
      new_col_name = f"CAGR_{init_year}_{end_year}_%"
      df[new_col_name] = growth_series.replace([np.inf, -np.inf], np.nan) * 100
      return df

##############################################################################
# Process groups data
##############################################################################
def __get_collaborators_data() -> pd.DataFrame:
   #Read collaborators data
   collaborators_df = pd.read_excel(COLABORATORS_XLSX, sheet_name='Sheet1',
                                    converters={'N.Identificación': str})
   collaborators_df = collaborators_df.rename(columns={'N.Identificación': 'IDENTIFICADOR'})
   collaborators_df['IDENTIFICADOR'] = collaborators_df['IDENTIFICADOR'].str.replace("'", '')
   return collaborators_df.drop_duplicates()

def __get_hired_inv() -> pd.DataFrame:
   hired_inv_df = pd.read_excel(INV_PROFILES_XLSX, 
                                    sheet_name='inv_ups',
                                    converters={'IDENTIFICADOR': str})
   hired_inv_df['IDENTIFICADOR'] = hired_inv_df['IDENTIFICADOR'].str.replace("'", '')
   #Filter hired researchers
   mask = hired_inv_df['CONTRATO_VIGENTE'].isin(['SI'])

   inv_gender_df = pd.read_excel(INV_PROFILES_XLSX,
                                    sheet_name='investigadores',
                                    converters={'IDENTIFICADOR': str})
   inv_gender_df['IDENTIFICADOR'] = inv_gender_df['IDENTIFICADOR'].str.replace("'", '')
   inv_gender_df = inv_gender_df[['IDENTIFICADOR', 'INV_GENERO', 'SEDE_INVESTIGADOR']].drop_duplicates()

   #Merge hired_inv_df with inv_gender
   hired_inv_df = pd.merge(hired_inv_df, inv_gender_df, on='IDENTIFICADOR', how='left')
   
   return hired_inv_df[mask].drop_duplicates()

def __get_affiliations_data(year: int = 2025) -> pd.DataFrame:
   #Read researchers data of affiliations
   inv_groups_df = pd.read_excel(SGI_STATS_XLSX, 
                                       sheet_name='researchers_groups',
                                       converters={'IDENTIFICADOR': str})
   inv_groups_df['IDENTIFICADOR'] = inv_groups_df['IDENTIFICADOR'].str.replace("'", '')
   #Filter by ANIO_VIGENTE, equals to year
   mask = (inv_groups_df['ANIO_VIGENTE'].isin([year]) &
         (inv_groups_df['ESTADO_GRUPO'].isin(['Activo'])))
   inv_groups_df = inv_groups_df[mask]
   #Remove EXTERNO from TIPO_ROL
   inv_groups_df = inv_groups_df[~inv_groups_df['TIPO_ROL'].isin(['EXTERNO'])]
   inv_groups_df['TIPO_ROL'] = inv_groups_df['TIPO_ROL'].str.capitalize()

   #Extract the columns of interest
   inv_groups_df = inv_groups_df[['SEDE_INVESTIGADOR', 'TII_DESCRIPCION', 
                                 'INV_CODIGO', 'IDENTIFICADOR', 'TIPO_ROL',
                                 'NOMBRE_COMPLETO', 'INV_GENERO']]
   
   #Drop duplicates
   inv_groups_df = inv_groups_df.drop_duplicates()
   #Replace INV_GENERO values
   inv_groups_df['INV_GENERO'] = inv_groups_df['INV_GENERO'].replace({
      'F': 'Femenino',
      'M': 'Masculino'
   })

   #Add collaborators info
   inv_groups_df = __add_collaborators_info(inv_groups_df)

   #Rename columns
   inv_groups_df = inv_groups_df.rename(columns={
      'SEDE_INVESTIGADOR': 'Sede',
      'TII_DESCRIPCION': 'Rol en el Grupo',
      'NOMBRE_COMPLETO': 'Docente Investigador',
      'INV_GENERO': 'Género',
      'TIPO_ROL': 'Rol en la UPS'
   })

   return inv_groups_df

def analyze_groups_data(year: int = 2025) -> pd.DataFrame:
   groups_data = []
   print("Previous year")
   #Read researchers and projects data
   inv_groups_df = __get_affiliations_data(year=year-1)
   
   #Count researchers by Genero
   inv_sede_type_title_prev_df = __count_unique_data_by_column(inv_groups_df, 
                                                         ['Género'], 
                                                         'IDENTIFICADOR', 'N.Investigadores')
   total_inv_prev = inv_sede_type_title_prev_df['N.Investigadores'].sum()
   groups_data.append(f"Total investigadores {year-1}: {total_inv_prev}")
   groups_data.append(inv_sede_type_title_prev_df.to_string(index=False))

   groups_data.append(LINE)
   print("Analyzing groups data...")
   #Read researchers and projects data
   inv_groups_df = __get_affiliations_data(year=year)
   
   #Count researchers by Genero
   inv_sede_type_title_df = __count_unique_data_by_column(inv_groups_df, 
                                                         ['Género'], 
                                                         'IDENTIFICADOR', 'N.Investigadores')
   total_inv = inv_sede_type_title_df['N.Investigadores'].sum()
   groups_data.append(f"Total investigadores {year}: {total_inv}")
   groups_data.append(inv_sede_type_title_df.to_string(index=False))

   #Calculate percentage growth
   inv_growth = __calculate_percentage_growth(total_inv_prev, total_inv)
   groups_data.append(f"Crecimiento porcentual de investigadores {year-1} a {year}: {inv_growth:.2f}%")

   inv_growth_df = pd.merge(inv_sede_type_title_prev_df, inv_sede_type_title_df, 
                           on='Género', suffixes=(f'_{year-1}', f'_{year}'))
   inv_growth_df = __calculate_percentage_growth_df(inv_growth_df, 
                                                      f'N.Investigadores_{year-1}', 
                                                      f'N.Investigadores_{year}')
   groups_data.append(f"Crecimiento porcentual de investigadores por género {year-1} a {year}:")
   groups_data.append(inv_growth_df.to_string(index=False))

   #Count researchers by Rol en la UPS
   inv_gen_rol_df = __count_unique_data_by_column(inv_groups_df, 
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
   inv_gender_title_df = __count_unique_data_by_column(inv_groups_df, 
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
   inv_sede_title_gender_df = __count_unique_data_by_column(inv_groups_df, 
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
   inv_sede_gender_title_df = __count_unique_data_by_column(inv_groups_df, 
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
   save_txt_file('\n'.join(groups_data), f'output/groups_data_{year}.txt')

##############################################################################
# Process areas data: projects and publications
###############################################################################

def __get_proy_data() -> pd.DataFrame:
   #Read all project data
   proy_df = pd.read_excel(SGI_STATS_XLSX,
                                 sheet_name='projects')
   proy_df['AREA_CACES'] = proy_df['AREA_CACES'].apply(
      lambda x: ' '.join(str(x).strip().split()) if pd.notnull(x) else x
   )
   return proy_df.drop_duplicates()

def __add_collaborators_info(df: pd.DataFrame) -> pd.DataFrame:
   #Read collaborators data
   collaborators_df = __get_collaborators_data()
   collaborators_df = collaborators_df[['IDENTIFICADOR', 
                                       'Nivel de Estudios',
                                       'Titulo']].drop_duplicates()
   #Merge df with collaborators_df on IDENTIFICADOR
   df = pd.merge(df, collaborators_df, on='IDENTIFICADOR', how='left')
   df['TIPO_ROL'] = df['TIPO_ROL'].str.capitalize()
   #Replace TIPO_ROL values
   df['TIPO_ROL'] = df['TIPO_ROL'].replace({
      'Docente por horas': 'Docente'
   })

   #Process Titulo column
   TITLE_FILTER = ['DOCTOR PH.D', 'MAGISTER']
   mask = ~df['Titulo'].isin(TITLE_FILTER) & df['Titulo'].notna()
   df['Titulo'] = np.where(mask, 'OTROS', df['Titulo'])
   df['Titulo'] = df['Titulo'].str.title()

   return df

def __fix_role_specifics(df: pd.DataFrame) -> pd.DataFrame:
   df['ROL_ESPECIFICO'] = df['TIPO_ROL'].fillna(df['ROL_ESPECIFICO'])
   df['ROL_ESPECIFICO'] = df['ROL_ESPECIFICO'].str.capitalize()
   #Replace values of ROL_ESPECIFICO
   df['ROL_ESPECIFICO'] = df['ROL_ESPECIFICO'].replace({
      'Docente por horas': 'Docente',
      'Est. de doctorado': 'Estudiante',
      'Est. de maestría': 'Estudiante',
      'Est. de grado': 'Estudiante',
      'Est. de pregrado': 'Estudiante'
   })
   #Fix Titulo column
   mask = df['Titulo'].isna() & (~df['ROL_ESPECIFICO'].isin(['Estudiante', 'Externo']))
   df.loc[mask, 'Titulo'] = 'Otros'
   return df

def __get_inv_proy_data() -> pd.DataFrame:
   #Read all inv proy data
   inv_proy_df = pd.read_excel(SGI_STATS_XLSX, 
                                 sheet_name='researchers_projects',
                                 converters={'IDENTIFICADOR': str})
   inv_proy_df['IDENTIFICADOR'] = inv_proy_df['IDENTIFICADOR'].str.replace("'", '')

   #Add collaborators info
   inv_proy_df = __add_collaborators_info(inv_proy_df)

   #Fix ROL_ESPECIFICO values
   inv_proy_df = __fix_role_specifics(inv_proy_df)

   return inv_proy_df.drop_duplicates()

def __get_publications_data() -> pd.DataFrame:
   #Read all publications data
   pub_df = pd.read_excel(SGI_STATS_XLSX,
                                 sheet_name='authors_publications',
                                 converters={'IDENTIFICADOR': str,
                                             'ANIO_PUBLICACION': int})
   pub_df['AREA_CONOCIMIENTO'] = pub_df['AREA_CONOCIMIENTO'].apply(
      lambda x: ' '.join(str(x).strip().split()) if pd.notnull(x) else x
   )
   pub_df['IDENTIFICADOR'] = pub_df['IDENTIFICADOR'].str.replace("'", '')

   #Add collaborators info
   pub_df = __add_collaborators_info(pub_df)
   #Fix ROL_ESPECIFICO values
   pub_df = __fix_role_specifics(pub_df)

   return pub_df.drop_duplicates()

def analyze_areas_data(year: int):
   areas_data = []
   areas_data.append(LINE)

   all_areas = set()

   #Analyze projects data
   #####################################################################################
   inv_proy_df = __get_inv_proy_data()

   #Filter by ANIO_VIGENTE equals to year
   mask = ((inv_proy_df['RANGO_INICIO'].isin([year]))
            & (~inv_proy_df['ROL_ESPECIFICO'].isin(['Externo'])))
   inv_proy_df = inv_proy_df[mask]
   #Count number of researchers in projects
   total_inv_title = inv_proy_df['IDENTIFICADOR'].nunique()
   areas_data.append(f"Total de investigadores que participaron en proyectos {year}: {total_inv_title}")

   #Count number of researchers by Titulo
   inv_proy_title_df = __count_unique_data_by_column(inv_proy_df,
                                                ['Titulo'],
                                                'IDENTIFICADOR',
                                                'N.Investigadores')
   total_inv_hired = inv_proy_title_df['N.Investigadores'].sum()
   areas_data.append(f"Total de investigadores por título académico en proyectos {year}: Total -> {total_inv_hired}")
   areas_data.append(inv_proy_title_df.to_string(index=False))
   all_areas.update(inv_proy_title_df['Titulo'].unique().tolist())

   #Get unique projects
   ids = inv_proy_df['CODIGO_PROYECTO'].unique().tolist()

   #Get project data
   proy_df = __get_proy_data()
   proy_df = proy_df[proy_df['CODIGO_PROYECTO'].isin(ids)]

   #Count number of projects by AREA_CACES
   proy_area_df = __count_unique_data_by_column(proy_df,
                                                ['AREA_CACES'],
                                                'CODIGO_PROYECTO',
                                                'N.Proyectos')
   total_proy = proy_df['CODIGO_PROYECTO'].nunique()
   areas_data.append(f"Total de proyectos por area de conocimiento {year}: Total -> {total_proy}")
   areas_data.append(proy_area_df.to_string(index=False))
   all_areas.update(proy_area_df['AREA_CACES'].unique().tolist())

   #Analyze publications data
   #####################################################################################
   pub_df = __get_publications_data()
   pub_df = pub_df[pub_df['ANIO_PUBLICACION'].isin([year])]
   pub_df = pub_df[~pub_df['ROL_ESPECIFICO'].isin(['Externo'])]
   pub_df = pub_df[pub_df['INV_CODIGO'].notna()]
   #Rename area column
   pub_df = pub_df.rename(columns={'AREA_CONOCIMIENTO': 'AREA_CACES'})

   #Count number of researchers in publications
   total_pub_inv = pub_df['IDENTIFICADOR'].nunique()
   areas_data.append(f"Total de investigadores que participaron en publicaciones {year}: {total_pub_inv}")

   #Count number of publications by Titulo
   pub_title_df = __count_unique_data_by_column(pub_df,
                                                ['Titulo'],
                                                'IDENTIFICADOR',
                                                'N.Investigadores')
   total_inv_hired_pub = pub_title_df['N.Investigadores'].sum()
   areas_data.append(f"Total de investigadores por título académico en publicaciones {year}: Total -> {total_inv_hired_pub}")
   areas_data.append(pub_title_df.to_string(index=False))
   all_areas.update(pub_title_df['Titulo'].unique().tolist())

   #Count number of publications by AREA_CACES
   pub_area_df = __count_unique_data_by_column(pub_df,
                                                ['AREA_CACES'],
                                                'PRO_CODIGO',
                                                'N.Publicaciones')
   total_pub = pub_df['PRO_CODIGO'].nunique()
   areas_data.append(f"Total publicaciones por area de conocimiento {year}: Total -> {total_pub}")
   areas_data.append(pub_area_df.to_string(index=False))
   all_areas.update(pub_area_df['AREA_CACES'].unique().tolist())
   
   #Analyze researchers into project and publication areas
   #####################################################################################
   proy_inv_area_df = inv_proy_df[['AREA_CACES', 'IDENTIFICADOR', 'ROL_ESPECIFICO', 'Titulo']].drop_duplicates()
   pub_inv_area_df = pub_df[['AREA_CACES', 'IDENTIFICADOR', 'ROL_ESPECIFICO', 'Titulo']].drop_duplicates()
   combined_inv_area_df = pd.concat([proy_inv_area_df, pub_inv_area_df]).drop_duplicates()
   
   #Sort by IDENTIFICADOR and ROL_ESPECIFICO ascending
   combined_inv_area_df = combined_inv_area_df.sort_values(by=['IDENTIFICADOR', 'ROL_ESPECIFICO'], ascending=[True, True])
   #Drop duplicates by IDENTIFICADOR, keep first (highest role)
   combined_inv_area_df = combined_inv_area_df.drop_duplicates(subset=['IDENTIFICADOR'], keep='first')

   #Count number of researchers by AREA_CACES
   inv_area_df = __count_unique_data_by_column(combined_inv_area_df,
                                                ['AREA_CACES'],
                                                'IDENTIFICADOR',
                                                'N.Investigadores')
   total_inv_area = combined_inv_area_df['IDENTIFICADOR'].nunique()
   areas_data.append(f"Total de investigadores que participaron en proyectos y/o publicaciones en el {year}: {total_inv_area}")
   areas_data.append(inv_area_df.to_string(index=False))
   all_areas.update(inv_area_df['AREA_CACES'].unique().tolist())
   #Count number of researchers by ROL_ESPECIFICO
   inv_role_area_df = __count_unique_data_by_column(combined_inv_area_df,
                                                ['ROL_ESPECIFICO'],
                                                'IDENTIFICADOR',
                                                'N.Investigadores')
   total_inv_role = inv_role_area_df['N.Investigadores'].sum()
   areas_data.append(f"Total de investigadores por rol específico en proyectos y/o publicaciones en el {year}: {total_inv_role}")
   areas_data.append(inv_role_area_df.to_string(index=False))
   #Count number of researchers by Titulo
   inv_title_area_df = __count_unique_data_by_column(combined_inv_area_df,
                                                ['Titulo'],
                                                'IDENTIFICADOR',
                                                'N.Investigadores')
   total_inv_title = inv_title_area_df['N.Investigadores'].sum()
   areas_data.append(f"Total de investigadores por título académico en proyectos y/o publicaciones en el {year}: {total_inv_title}")
   areas_data.append(inv_title_area_df.to_string(index=False))
   all_areas.update(inv_title_area_df['Titulo'].unique().tolist())

   #Create a DataFrame with all areas
   all_areas_df = pd.DataFrame({'AREA_CACES': list(all_areas)}).drop_duplicates()
   #Merge with proy_area_df, pub_area_df and inv_area_df
   merged_areas_df = (all_areas_df.merge(proy_area_df[['AREA_CACES', 'N.Proyectos']], on='AREA_CACES', how='left')
                                 .merge(pub_area_df[['AREA_CACES', 'N.Publicaciones']], on='AREA_CACES', how='left')
                                 .merge(inv_area_df[['AREA_CACES', 'N.Investigadores']], on='AREA_CACES', how='left')
                                 .fillna(0))
   merged_areas_df['Total'] = (merged_areas_df['N.Proyectos']*0.35 +
                              merged_areas_df['N.Publicaciones']*0.25 +
                              merged_areas_df['N.Investigadores']*0.4)
   #Sort by Total descending
   merged_areas_df = merged_areas_df.sort_values(by='Total', ascending=False)
   #Remove any N.Proyectos, N.Publicaciones, N.Investigadores equals to 0
   mask = (merged_areas_df['N.Proyectos'] > 0) & (merged_areas_df['N.Publicaciones'] > 0) & (merged_areas_df['N.Investigadores'] > 0)
   merged_areas_df = merged_areas_df[mask]
   areas_data.append(LINE)
   areas_data.append("Resumen de áreas CACES involucradas en proyectos y publicaciones con participación de investigadores:")
   areas_data.append(merged_areas_df.to_string(index=False))
   #Plot scatter plot
   ch.plot_scatter_chart_broken_axis(merged_areas_df,
                        x_col='N.Proyectos',
                        y_col='N.Publicaciones',
                        size_col='N.Investigadores',
                        label_col='AREA_CACES',
                        y_break_low=45,
                        y_break_high=90,
                        label_x_threshold=4,
                        label_y_threshold=10)

   #Print areas data
   print('\n'.join(areas_data))
   #Save data to txt file
   save_txt_file('\n'.join(areas_data), f'output/areas_data_{year}.txt')

##############################################################################
# Process publications data
##############################################################################

def __get_all_publications_data() -> pd.DataFrame:
   #Read all publications data
   pub_df = pd.read_excel(ALL_PURE_PUBLICATIONS_XLSX,
                           sheet_name='Sheet0',
                           converters={'CODIGO_PURE': str})
   #Filter valid publications types
   VALID_PUB_TYPES = {'Article': 'Artículo de revista',
                     'Chapter': 'Capítulo de libro',
                     'Conference contribution': 'Artículo de conferencia',
                     'Paper': 'Artículo de conferencia',
                     'Book': 'Libro',
                     'Review article': 'Artículo de revista',
                     'Conference article': 'Artículo de conferencia',
                     'Letter': 'Artículo de revista',
                     'Literature review': 'Artículo de revista',
                     }

   mask = pub_df['TIPO_PRODUCTO'].isin(VALID_PUB_TYPES.keys())
   pub_df = pub_df[mask].drop_duplicates()
   #Replace TIPO_PRODUCTO values
   pub_df['TIPO_PRODUCTO'] = pub_df['TIPO_PRODUCTO'].replace(VALID_PUB_TYPES)

   return pub_df.drop_duplicates()

def __get_sjr_jcr_publications_data() -> pd.DataFrame:
   #Read SJR and JCR publications data
   sjr_jcr_df = pd.read_excel(ALL_SJR_JCR_PUBLICATIONS_XLSX,
                           sheet_name='all_publications',
                           converters={'CODIGO_PURE': str})
   return sjr_jcr_df.drop_duplicates()

def __get_scopus_citations_data() -> pd.DataFrame:
   #Read researchers Scopus citations data
   citations_df = pd.read_excel(INV_CITATIONS_SCOPUS_XLSX, 
                                 sheet_name='Sheet0',
                                 converters={'IDENTIFICADOR': str})
   citations_df['IDENTIFICADOR'] = citations_df['IDENTIFICADOR'].str.replace("'", '')
   return citations_df.drop_duplicates()

def __read_scopus_data() -> pd.DataFrame:
   #Read Scopus data
   scopus_df = pd.read_excel(SCOPUS_PUB_XLSX,
                           sheet_name='scopus',
                           converters={'EID': str})
   return scopus_df.drop_duplicates()

def __read_wos_data() -> pd.DataFrame:
   #Read Web of Science data
   wos_df = pd.read_excel(WOS_PUB_XLSX,
                           sheet_name='savedrecs',
                           converters={'UT (Unique WOS ID)': str})
   return wos_df.drop_duplicates()

def analyze_publications_data(year: int):
   pub_data = []
   pub_data.append(LINE)

   #Analyze publications data
   #####################################################################################
   #Get all publications data
   all_pure_pub_df = __get_all_publications_data()

   #Get current year publications
   current_year_pub_df = all_pure_pub_df[all_pure_pub_df['ANIO_PUBLICACION'].isin([year])]
   total_current_year_pub = current_year_pub_df['CODIGO_PURE'].nunique()
   pub_data.append(f"Total de publicaciones en el año {year}: {total_current_year_pub}")

   #Previous year publications
   previous_year_pub_df = all_pure_pub_df[all_pure_pub_df['ANIO_PUBLICACION'].isin([year-1])]
   total_previous_year_pub = previous_year_pub_df['CODIGO_PURE'].nunique()
   pub_data.append(f"Total de publicaciones en el año {year-1}: {total_previous_year_pub}")

   #Calculate percentage growth
   pub_growth = __calculate_percentage_growth(total_previous_year_pub, total_current_year_pub)
   pub_data.append(f"Crecimiento porcentual de publicaciones de {year-1} a {year}: {pub_growth:.2f}%")

   #Count publications by TIPO_PRODUCTO, current year
   pub_type_current_df = __count_unique_data_by_column(current_year_pub_df,
                                                ['TIPO_PRODUCTO'],
                                                'CODIGO_PURE',
                                                'N.Publicaciones')
   total_pub_type = pub_type_current_df['N.Publicaciones'].sum()
   pub_data.append(f"Total de publicaciones por tipo de producto en el año {year}: Total -> {total_pub_type}")
   pub_data.append(pub_type_current_df.to_string(index=False))

   #Count publications by TIPO_PRODUCTO, previous year
   pub_type_prev_df = __count_unique_data_by_column(previous_year_pub_df,
                                                ['TIPO_PRODUCTO'],
                                                'CODIGO_PURE',
                                                'N.Publicaciones')
   total_pub_type_prev = pub_type_prev_df['N.Publicaciones'].sum()
   pub_data.append(f"Total de publicaciones por tipo de producto en el año {year-1}: Total -> {total_pub_type_prev}")
   pub_data.append(pub_type_prev_df.to_string(index=False))

   #Calculate percentage growth by TIPO_PRODUCTO
   pub_type_growth_df = pd.merge(pub_type_prev_df, pub_type_current_df, 
                           on='TIPO_PRODUCTO', suffixes=(f'_{year-1}', f'_{year}'))
   pub_type_growth_df = __calculate_percentage_growth_df(pub_type_growth_df, 
                                                      f'N.Publicaciones_{year-1}', 
                                                      f'N.Publicaciones_{year}')
   pub_data.append(f"Crecimiento porcentual de publicaciones por tipo de producto de {year-1} a {year}:")
   pub_data.append(pub_type_growth_df.to_string(index=False))

      #Analyze scopus and wos data
   #####################################################################################
   #Process scopus data
   scopus_df = __read_scopus_data()
   #Filter scopus by range
   mask = scopus_df['Year'].between(2021, 2025)
   range_scopus_df = scopus_df[mask]
   #Count publications by Year
   scopus_year_df = __count_unique_data_by_column(range_scopus_df,
                                                ['Year'],
                                                'EID',
                                                'N.Publicaciones')
   #Transpose dataframe
   scopus_year_df = scopus_year_df.pivot_table(index=None,
                                                columns='Year',
                                                values='N.Publicaciones',
                                                fill_value=0).reset_index()
   scopus_year_df = __calculate_percentage_growth_df(scopus_year_df, 2024, 2025)
   scopus_year_df = __calcula_anual_percentage_growth_df(scopus_year_df, 2021, 2025)
   pub_data.append(LINE)
   pub_data.append("Distribución de publicaciones en Scopus por año:")
   pub_data.append(scopus_year_df.to_string(index=False))

   #Process Web of Science data
   wos_df = __read_wos_data()
   #Filter wos by range
   mask = wos_df['Publication Year'].between(2021, 2025)
   range_wos_df = wos_df[mask]
   #Count publications by Publication Year
   wos_year_df = __count_unique_data_by_column(range_wos_df,
                                                ['Publication Year'],
                                                'UT (Unique WOS ID)',
                                                'N.Publicaciones')
   #Transpose dataframe
   wos_year_df = wos_year_df.pivot_table(index=None,
                                          columns='Publication Year',
                                          values='N.Publicaciones',
                                          fill_value=0).reset_index()
   wos_year_df = __calculate_percentage_growth_df(wos_year_df, 2024, 2025)
   wos_year_df = __calcula_anual_percentage_growth_df(wos_year_df, 2021, 2025)
   pub_data.append(LINE)
   pub_data.append("Distribución de publicaciones en Web of Science por año:")
   pub_data.append(wos_year_df.to_string(index=False))

   #Analyze SJR and JCR data
   #####################################################################################
   #Get SJR and JCR data
   all_pub_df = __get_all_publications_data()
   #Filter year_range between 2019 to 2025
   mask = all_pub_df['ANIO_PUBLICACION'].between(2021, 2025)
   all_pub_df = all_pub_df[mask]
   #Get all ids
   ids = all_pub_df['CODIGO_PURE'].unique().tolist()
   #Get SJR and JCR data
   sjr_jcr_df = __get_sjr_jcr_publications_data()
   #Filter by ids
   range_sjr_jcr_df = sjr_jcr_df[sjr_jcr_df['CODIGO_PURE'].isin(ids)]

   #Get SJR data for range years
   range_sjr_df = range_sjr_jcr_df[range_sjr_jcr_df['SJR Best Quartile'].isin(['Q1', 'Q2', 'Q3', 'Q4'])]
   #Pivot table by SJR and ANIO_PUBLICACION
   sjr_pivot_df = (range_sjr_df.pivot_table(index='SJR Best Quartile',
                                          columns='ANIO_PUBLICACION',
                                          values='CODIGO_PURE',
                                          aggfunc='nunique',
                                          fill_value=0).reset_index())
   pub_data.append(LINE)
   #Plot line chart
   ch.plot_lines_chart(sjr_pivot_df.copy(),
                     category_col='SJR Best Quartile',
                     y_label='N. Publicaciones',
                     )
   sjr_growth_df = __calculate_percentage_growth_df(sjr_pivot_df, 2024, 2025)
   sjr_growth_df = __calcula_anual_percentage_growth_df(sjr_growth_df, 2021, 2025)
   pub_data.append(f"""Distribución de publicaciones por SJR y año de publicación y 
                     Crecimiento porcentual de publicaciones por SJR Best Quartile de 2024 a 2025:""")
   pub_data.append(sjr_growth_df.to_string(index=False))

   #Get JCR data for range years
   range_jcr_df = range_sjr_jcr_df[range_sjr_jcr_df['JIF Quartile'].isin(['Q1', 'Q2', 'Q3', 'Q4'])]
   #Pivot table by JCR and ANIO_PUBLICACION
   jcr_pivot_df = (range_jcr_df.pivot_table(index='JIF Quartile',
                                          columns='ANIO_PUBLICACION',
                                          values='CODIGO_PURE',
                                          aggfunc='nunique',
                                          fill_value=0).reset_index())
   pub_data.append(LINE)
   #Plot line chart
   ch.plot_lines_chart(jcr_pivot_df.copy(),
                     category_col='JIF Quartile',
                     y_label='N. Publicaciones')
   jcr_growth_df = __calculate_percentage_growth_df(jcr_pivot_df, 2024, 2025)
   jcr_growth_df = __calcula_anual_percentage_growth_df(jcr_growth_df, 2021, 2025)
   pub_data.append(f"""Distribución de publicaciones por JCR y año de publicación y 
                     Crecimiento porcentual de publicaciones por JCR Quartile de 2024 a 2025:""")
   pub_data.append(jcr_growth_df.to_string(index=False))

   #Analyze citations data
   #####################################################################################
   #Get Scopus citations data
   citations_df = __get_scopus_citations_data()
   #Keep relevant columns
   citations_df = citations_df[['CODIGO_PURE', 'ANIO', 'CITAS']].drop_duplicates()
   #Filter by ANIO between 2021 and 2025
   mask = citations_df['ANIO'].between(2021, 2025)
   range_citations_df = citations_df[mask]
   #Sum CITAS by ANIO
   citations_year_df = (range_citations_df.groupby('ANIO')
                                          .agg({'CITAS': 'sum'})
                                          .reset_index())
   #Transpose dataframe
   citations_year_df = citations_year_df.pivot_table(index=None,
                                                columns='ANIO',
                                                values='CITAS',
                                                fill_value=0).reset_index()
   print(citations_year_df)
   citations_year_df = __calculate_percentage_growth_df(citations_year_df, 2024.0, 2025.0)
   citations_year_df = __calcula_anual_percentage_growth_df(citations_year_df, 2021.0, 2025.0)
   pub_data.append(LINE)
   pub_data.append("Distribución de citas en Scopus por año:")
   pub_data.append(citations_year_df.to_string(index=False))

   #Save data to txt file
   print('\n'.join(pub_data))
   save_txt_file('\n'.join(pub_data), f'output/publications_data_{year}.txt')

##############################################################################
# Process projects data
##############################################################################

def analyze_projects_data(year: int):
   #Analyze projects data
   #####################################################################################
   proy_data = []
   proy_data.append(LINE)
   #Get all project data
   proy_df = __get_proy_data()
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
   proy_growth = __calculate_percentage_growth(total_previous_proy, total_current_proy)
   proy_data.append(f"Crecimiento porcentual de proyectos de {year-1} a {year}: {proy_growth:.2f}%")

   #Count number of projects by Sede
   proy_sede_df = __count_unique_data_by_column(current_proy_df,
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
   proy_ods_df = __count_unique_data_by_column(current_proy_df,
                                                ['Objetivos de Desarrollo Sostenible (ODS)'],
                                                'CODIGO_PROYECTO',
                                                'N.Proyectos')
   total_proy_ods = proy_ods_df['N.Proyectos'].sum()
   proy_data.append(f"Total de proyectos por ODS en el año {year}: Total -> {total_proy_ods}")
   proy_data.append(proy_ods_df.to_string(index=False))
   ch.plot_horizontal_bar_chart(proy_ods_df,
                              category_col='Objetivos de Desarrollo Sostenible (ODS)',
                              value_col='N.Proyectos')
   
   #Count number of projects by PROGRAMA_VINCULACION
   proy_prog_df = __count_unique_data_by_column(current_proy_df,
                                                ['Programa de Vinculación'],
                                                'CODIGO_PROYECTO',
                                                'N.Proyectos')
   total_proy_prog = proy_prog_df['N.Proyectos'].sum()
   proy_data.append(f"Total de proyectos por programa de vinculación en el año {year}: Total -> {total_proy_prog}")
   proy_data.append(proy_prog_df.to_string(index=False))
   ch.plot_horizontal_bar_chart(proy_prog_df,
                              category_col='Programa de Vinculación',
                              value_col='N.Proyectos')
   
   #Print projects data
   print('\n'.join(proy_data))
   #Save data to txt file
   save_txt_file('\n'.join(proy_data), f'output/projects_data_{year}.txt')

if __name__ == '__main__':
   #Step 1: Analyze groups data
   #analyze_groups_data(year=2025)
   #Step 2: Analyze areas data
   analyze_areas_data(year=2025)
   #Step 3: Analyze publications
   #analyze_publications_data(year=2025)
   #Step 4: Analyze projects
   #analyze_projects_data(year=2025)