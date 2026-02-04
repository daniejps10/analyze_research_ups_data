import pandas as pd
import numpy as np
from src.config import settings

##############################################################################
# LOADERS
##############################################################################

def get_groups_data() -> pd.DataFrame:
   #Read all groups data
   groups_df = pd.read_excel(settings.SGI_GROUPS_XLSX, 
                           sheet_name='grupos')
   #Drop duplicates
   return groups_df.drop_duplicates()

def get_collaborators_data() -> pd.DataFrame:
   #Read collaborators data
   collaborators_df = pd.read_excel(settings.COLABORATORS_XLSX, sheet_name='Sheet1',
                                    converters={'N.Identificación': str})
   collaborators_df = collaborators_df.rename(columns={'N.Identificación': 'IDENTIFICADOR'})
   collaborators_df['IDENTIFICADOR'] = collaborators_df['IDENTIFICADOR'].str.replace("'", '')
   return collaborators_df.drop_duplicates()

def get_hired_inv() -> pd.DataFrame:
   hired_inv_df = pd.read_excel(settings.INV_PROFILES_XLSX, 
                                    sheet_name='inv_ups',
                                    converters={'IDENTIFICADOR': str})
   hired_inv_df['IDENTIFICADOR'] = hired_inv_df['IDENTIFICADOR'].str.replace("'", '')
   #Filter hired researchers
   mask = hired_inv_df['CONTRATO_VIGENTE'].isin(['SI'])

   inv_gender_df = pd.read_excel(settings.INV_PROFILES_XLSX,
                                    sheet_name='investigadores',
                                    converters={'IDENTIFICADOR': str})
   inv_gender_df['IDENTIFICADOR'] = inv_gender_df['IDENTIFICADOR'].str.replace("'", '')
   inv_gender_df = inv_gender_df[['IDENTIFICADOR', 'INV_GENERO', 'SEDE_INVESTIGADOR']].drop_duplicates()

   #Merge hired_inv_df with inv_gender
   hired_inv_df = pd.merge(hired_inv_df, inv_gender_df, on='IDENTIFICADOR', how='left')
   
   return hired_inv_df[mask].drop_duplicates()

def add_collaborators_info(df: pd.DataFrame) -> pd.DataFrame:
   #Read collaborators data
   collaborators_df = get_collaborators_data()
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

def get_affiliations_data(year: int = 2025) -> pd.DataFrame:
   #Read researchers data of affiliations
   inv_groups_df = pd.read_excel(settings.SGI_STATS_XLSX, 
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
   inv_groups_df = add_collaborators_info(inv_groups_df)

   #Rename columns
   inv_groups_df = inv_groups_df.rename(columns={
      'SEDE_INVESTIGADOR': 'Sede',
      'TII_DESCRIPCION': 'Rol en el Grupo',
      'NOMBRE_COMPLETO': 'Docente Investigador',
      'INV_GENERO': 'Género',
      'TIPO_ROL': 'Rol en la UPS'
   })

   return inv_groups_df

def get_proy_data() -> pd.DataFrame:
   #Read all project data
   proy_df = pd.read_excel(settings.SGI_STATS_XLSX,
                                 sheet_name='projects')
   proy_df['AREA_CACES'] = proy_df['AREA_CACES'].apply(
      lambda x: ' '.join(str(x).strip().split()) if pd.notnull(x) else x
   )
   return proy_df.drop_duplicates()

def fix_role_specifics(df: pd.DataFrame) -> pd.DataFrame:
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

def get_inv_proy_data() -> pd.DataFrame:
   #Read all inv proy data
   inv_proy_df = pd.read_excel(settings.SGI_STATS_XLSX, 
                                 sheet_name='researchers_projects',
                                 converters={'IDENTIFICADOR': str})
   inv_proy_df['IDENTIFICADOR'] = inv_proy_df['IDENTIFICADOR'].str.replace("'", '')

   #Add collaborators info
   inv_proy_df = add_collaborators_info(inv_proy_df)

   #Fix ROL_ESPECIFICO values
   inv_proy_df = fix_role_specifics(inv_proy_df)

   return inv_proy_df.drop_duplicates()

def get_sgi_publications_data() -> pd.DataFrame:
   #Read all publications data
   pub_df = pd.read_excel(settings.SGI_STATS_XLSX,
                                 sheet_name='authors_publications',
                                 converters={'IDENTIFICADOR': str,
                                             'CODIGO_SCOPUS': str,
                                             'ANIO_PUBLICACION': int})
   pub_df['AREA_CONOCIMIENTO'] = pub_df['AREA_CONOCIMIENTO'].apply(
      lambda x: ' '.join(str(x).strip().split()) if pd.notnull(x) else x
   )
   pub_df['IDENTIFICADOR'] = pub_df['IDENTIFICADOR'].str.replace("'", '')
   pub_df['CODIGO_SCOPUS'] = pub_df['CODIGO_SCOPUS'].str.replace("'", '')

   #Add collaborators info
   pub_df = add_collaborators_info(pub_df)
   #Fix ROL_ESPECIFICO values
   pub_df = fix_role_specifics(pub_df)

   return pub_df.drop_duplicates()

def get_all_pure_publications_data() -> pd.DataFrame:
   #Read all publications data
   pub_df = pd.read_excel(settings.ALL_PURE_PUBLICATIONS_XLSX,
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
   #Change campus
   campus_mapping = {
      'org2': 'Sede Quito',
      'org3': 'Sede Guayaquil',
      'org4': 'Sede Cuenca'
   }

   # Update only the rows that match your codes
   pub_df['SEDE_ADMINISTRADOR'] = (pub_df['GRU_CODIGO']
                                 .map(campus_mapping).fillna(pub_df['SEDE_ADMINISTRADOR']))

   return pub_df.drop_duplicates()

def get_sjr_jcr_publications_data() -> pd.DataFrame:
   #Read SJR and JCR publications data
   sjr_jcr_df = pd.read_excel(settings.ALL_SJR_JCR_PUBLICATIONS_XLSX,
                           sheet_name='all_publications',
                           converters={'CODIGO_PURE': str})
   return sjr_jcr_df.drop_duplicates()

def get_scopus_citations_data() -> pd.DataFrame:
   #Read researchers Scopus citations data
   citations_df = pd.read_excel(settings.INV_CITATIONS_SCOPUS_XLSX, 
                                 sheet_name='Sheet0',
                                 converters={'IDENTIFICADOR': str})
   citations_df['IDENTIFICADOR'] = citations_df['IDENTIFICADOR'].str.replace("'", '')
   return citations_df.drop_duplicates()

def read_scopus_data() -> pd.DataFrame:
   #Read Scopus data
   scopus_df = pd.read_excel(settings.SCOPUS_PUB_XLSX,
                           sheet_name='scopus',
                           converters={'EID': str})
   return scopus_df.drop_duplicates()

def read_wos_data() -> pd.DataFrame:
   #Read Web of Science data
   wos_df = pd.read_excel(settings.WOS_PUB_XLSX,
                           sheet_name='savedrecs',
                           converters={'UT (Unique WOS ID)': str})
   return wos_df.drop_duplicates()

def get_pub_groups_data() -> pd.DataFrame:
   #Process data from SCOPUS publications
   #--------------------------------------------------------------------------------------
   all_pure_pub_data = get_all_pure_publications_data()
   #Keep relevant columns
   all_pure_pub_data = all_pure_pub_data[['CODIGO_PURE',
                                          'GRU_CODIGO',
                                          'SEDE_ADMINISTRADOR']].drop_duplicates()
   #Get all pure ids
   pure_ids = all_pure_pub_data['CODIGO_PURE'].unique().tolist()

   #Process data from SGI publications
   #--------------------------------------------------------------------------------------
   #Read all publications groups data
   sgi_pub_groups_df = get_sgi_publications_data()
   #Keep relevant columns
   sgi_pub_groups_df = sgi_pub_groups_df[['PRO_CODIGO',
                                    'CODIGO_SCOPUS',
                                    'GRUPO_CODIGO_ADMINISTRADOR',
                                    'GRUPO_ADMINISTRADOR',
                                    'SEDE_ADMINISTRADOR']].drop_duplicates()
   #Map campus
   campus_dict = {
      'QUITO': 'Sede Quito',
      'GUAYAQUIL': 'Sede Guayaquil',
      'MATRIZ CUENCA': 'Sede Cuenca'
   }
   sgi_pub_groups_df['SEDE_ADMINISTRADOR'] = sgi_pub_groups_df['SEDE_ADMINISTRADOR'].map(campus_dict)
   #Create new columns
   sgi_pub_groups_df['TEMP_CODE'] = 'ro' + sgi_pub_groups_df['PRO_CODIGO'].astype(str)
   sgi_pub_groups_df['CODIGO_SCOPUS'] = sgi_pub_groups_df['CODIGO_SCOPUS'].str.replace("2-s2.0-", '', regex=False)
   sgi_pub_groups_df['CODIGO_PURE'] = sgi_pub_groups_df['CODIGO_SCOPUS'].fillna(sgi_pub_groups_df['TEMP_CODE'])
   sgi_pub_groups_df['GRU_CODIGO'] = 'orgsgi' + sgi_pub_groups_df['GRUPO_CODIGO_ADMINISTRADOR'].astype(str)
   #Keep relevant columns
   sgi_pub_groups_df = sgi_pub_groups_df[['CODIGO_PURE',
                                    'GRU_CODIGO',
                                    'SEDE_ADMINISTRADOR']].drop_duplicates()
   #Filter by pure_ids
   filter_sgi_pub_groups_df = sgi_pub_groups_df[sgi_pub_groups_df['CODIGO_PURE'].isin(pure_ids)]
   sgi_ids = filter_sgi_pub_groups_df['CODIGO_PURE'].unique().tolist()

   #Filter all_pure_pub_data by sgi_ids
   filter_pure_pub_data = all_pure_pub_data[~all_pure_pub_data['CODIGO_PURE'].isin(sgi_ids)]
   
   #Combine both dataframes
   pub_groups_df = pd.concat([filter_sgi_pub_groups_df, filter_pure_pub_data]).drop_duplicates()

   #Add legend group data
   groups_df = get_groups_data()
   groups_df = groups_df[['PURE_ORG_ID', 'GRU_LEYENDA']].drop_duplicates()
   groups_df = groups_df.rename(columns={'PURE_ORG_ID': 'GRU_CODIGO',
                                       'GRU_LEYENDA': 'GRUPO_LEYENDA'})
   pub_groups_df = pub_groups_df.merge(groups_df, on='GRU_CODIGO', how='left')
   return pub_groups_df.drop_duplicates()

def get_pure_caces_data_type_df() -> pd.DataFrame:
   #Read PURE CACES data type
   pure_caces_df = pd.read_excel(settings.ALL_SJR_JCR_PUBLICATIONS_XLSX,
                                 sheet_name='caces_pub_proy',
                                 converters={'CODIGO_PURE': str})
   pure_caces_df['CODIGO_PURE'] = pure_caces_df['CODIGO_PURE'].str.replace("'", '')
   return pure_caces_df.drop_duplicates()

def get_sival_institutions_data() -> pd.DataFrame:
   #Read SciVal institutions data
   scival_inst_df = pd.read_excel(settings.SCIVAL_INSTITUTIONS_XLSX,
                           sheet_name='instituciones')
   return scival_inst_df.drop_duplicates()

def get_all_pub_aut_org_data() -> pd.DataFrame:
   #Read all institutions data
   pub_aut_org_df = pd.read_excel(settings.PURE_AUT_ORG_XLSX,
                           sheet_name='Sheet0',
                           converters={'CODIGO_PURE': str})
   pub_aut_org_df['CODIGO_PURE'] = pub_aut_org_df['CODIGO_PURE'].str.replace("'", '', regex=False)
   return pub_aut_org_df.drop_duplicates()
