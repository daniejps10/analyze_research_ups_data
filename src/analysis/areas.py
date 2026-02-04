import pandas as pd
from src.utils import util
from src import charts as ch
from src.data import loader
from src.config import settings

def analyze_areas_data(year: int):
   areas_data = []
   areas_data.append(settings.LINE)
   all_areas = set()

   #Analyze projects data
   #####################################################################################
   inv_proy_df = loader.get_inv_proy_data()

   #Filter by ANIO_VIGENTE equals to year
   mask = ((inv_proy_df['RANGO_INICIO'].isin([year]))
            & (~inv_proy_df['ROL_ESPECIFICO'].isin(['Externo'])))
   inv_proy_df = inv_proy_df[mask]
   #Count number of researchers in projects
   total_inv_title = inv_proy_df['IDENTIFICADOR'].nunique()
   areas_data.append(f"Total de investigadores que participaron en proyectos {year}: {total_inv_title}")

   #Count number of researchers by Titulo
   inv_proy_title_df = util.count_unique_data_by_column(inv_proy_df,
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
   proy_df = loader.get_proy_data()
   proy_df = proy_df[proy_df['CODIGO_PROYECTO'].isin(ids)]

   #Count number of projects by AREA_CACES
   proy_area_df = util.count_unique_data_by_column(proy_df,
                                                ['AREA_CACES'],
                                                'CODIGO_PROYECTO',
                                                'N.Proyectos')
   total_proy = proy_df['CODIGO_PROYECTO'].nunique()
   areas_data.append(f"Total de proyectos por area de conocimiento {year}: Total -> {total_proy}")
   areas_data.append(proy_area_df.to_string(index=False))
   all_areas.update(proy_area_df['AREA_CACES'].unique().tolist())

   #Analyze publications data
   #####################################################################################
   pub_df = loader.get_sgi_publications_data()
   pub_df = pub_df[pub_df['ANIO_PUBLICACION'].isin([year])]
   pub_df = pub_df[~pub_df['ROL_ESPECIFICO'].isin(['Externo'])]
   pub_df = pub_df[pub_df['INV_CODIGO'].notna()]
   #Rename area column
   pub_df = pub_df.rename(columns={'AREA_CONOCIMIENTO': 'AREA_CACES'})

   #Count number of researchers in publications
   total_pub_inv = pub_df['IDENTIFICADOR'].nunique()
   areas_data.append(f"Total de investigadores que participaron en publicaciones {year}: {total_pub_inv}")

   #Count number of publications by Titulo
   pub_title_df = util.count_unique_data_by_column(pub_df,
                                                ['Titulo'],
                                                'IDENTIFICADOR',
                                                'N.Investigadores')
   total_inv_hired_pub = pub_title_df['N.Investigadores'].sum()
   areas_data.append(f"Total de investigadores por título académico en publicaciones {year}: Total -> {total_inv_hired_pub}")
   areas_data.append(pub_title_df.to_string(index=False))
   all_areas.update(pub_title_df['Titulo'].unique().tolist())

   #Count number of publications by AREA_CACES
   pub_area_df = util.count_unique_data_by_column(pub_df,
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
   inv_area_df = util.count_unique_data_by_column(combined_inv_area_df,
                                                ['AREA_CACES'],
                                                'IDENTIFICADOR',
                                                'N.Investigadores')
   total_inv_area = combined_inv_area_df['IDENTIFICADOR'].nunique()
   areas_data.append(f"Total de investigadores que participaron en proyectos y/o publicaciones en el {year}: {total_inv_area}")
   areas_data.append(inv_area_df.to_string(index=False))
   all_areas.update(inv_area_df['AREA_CACES'].unique().tolist())
   #Count number of researchers by ROL_ESPECIFICO
   inv_role_area_df = util.count_unique_data_by_column(combined_inv_area_df,
                                                ['ROL_ESPECIFICO'],
                                                'IDENTIFICADOR',
                                                'N.Investigadores')
   total_inv_role = inv_role_area_df['N.Investigadores'].sum()
   areas_data.append(f"Total de investigadores por rol específico en proyectos y/o publicaciones en el {year}: {total_inv_role}")
   areas_data.append(inv_role_area_df.to_string(index=False))
   #Count number of researchers by Titulo
   inv_title_area_df = util.count_unique_data_by_column(combined_inv_area_df,
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
   areas_data.append(settings.LINE)
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
   util.save_txt_file('\n'.join(areas_data), f'output/areas_data_{year}.txt')
