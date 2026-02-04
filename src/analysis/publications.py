import pandas as pd
from src.utils import util
from src import charts as ch
from src.data import loader
from src.config import settings

def analyze_publications_data(year: int):

   print(settings.LINE)
   print("Analyzing publications data...")
   print(settings.LINE)

   pub_data = []
   pub_data.append(settings.LINE)

   #Analyze publications data
   #####################################################################################
   #Get all publications data
   all_pure_pub_df = loader.get_all_pure_publications_data()

   #Get current year publications
   current_year_pub_df = all_pure_pub_df[all_pure_pub_df['ANIO_PUBLICACION'].isin([year])]
   total_current_year_pub = current_year_pub_df['CODIGO_PURE'].nunique()
   pub_data.append(f"Total de publicaciones en el año {year}: {total_current_year_pub}")

   #Previous year publications
   previous_year_pub_df = all_pure_pub_df[all_pure_pub_df['ANIO_PUBLICACION'].isin([year-1])]
   total_previous_year_pub = previous_year_pub_df['CODIGO_PURE'].nunique()
   pub_data.append(f"Total de publicaciones en el año {year-1}: {total_previous_year_pub}")

   #Calculate percentage growth
   pub_growth = util.calculate_percentage_growth(total_previous_year_pub, total_current_year_pub)
   pub_data.append(f"Crecimiento porcentual de publicaciones de {year-1} a {year}: {pub_growth:.2f}%")

   #Count publications by TIPO_PRODUCTO, current year
   pub_type_current_df = util.count_unique_data_by_column(current_year_pub_df,
                                                ['TIPO_PRODUCTO'],
                                                'CODIGO_PURE',
                                                'N.Publicaciones')
   total_pub_type = pub_type_current_df['N.Publicaciones'].sum()
   pub_data.append(f"Total de publicaciones por tipo de producto en el año {year}: Total -> {total_pub_type}")
   pub_data.append(pub_type_current_df.to_string(index=False))

   #Count publications by TIPO_PRODUCTO, previous year
   pub_type_prev_df = util.count_unique_data_by_column(previous_year_pub_df,
                                                ['TIPO_PRODUCTO'],
                                                'CODIGO_PURE',
                                                'N.Publicaciones')
   total_pub_type_prev = pub_type_prev_df['N.Publicaciones'].sum()
   pub_data.append(f"Total de publicaciones por tipo de producto en el año {year-1}: Total -> {total_pub_type_prev}")
   pub_data.append(pub_type_prev_df.to_string(index=False))
   ch.plot_pie_chart(pub_type_current_df.copy(),
                     category_col='TIPO_PRODUCTO',
                     value_col='N.Publicaciones')

   #Calculate percentage growth by TIPO_PRODUCTO
   pub_type_growth_df = pd.merge(pub_type_prev_df, pub_type_current_df, 
                           on='TIPO_PRODUCTO', suffixes=(f'_{year-1}', f'_{year}'))
   pub_type_growth_df = util.calculate_percentage_growth_df(pub_type_growth_df, 
                                                      f'N.Publicaciones_{year-1}', 
                                                      f'N.Publicaciones_{year}')
   pub_data.append(f"Crecimiento porcentual de publicaciones por tipo de producto de {year-1} a {year}:")
   pub_data.append(pub_type_growth_df.to_string(index=False))

      #Analyze scopus and wos data
   #####################################################################################
   #Process scopus data
   scopus_df = loader.read_scopus_data()
   #Filter scopus by range
   mask = scopus_df['Year'].between(2021, 2025)
   range_scopus_df = scopus_df[mask]
   #Count publications by Year
   scopus_year_df = util.count_unique_data_by_column(range_scopus_df,
                                                ['Year'],
                                                'EID',
                                                'N.Publicaciones')
   ch.plot_lines_chart(scopus_year_df.copy(),
                     x_col='Year',
                     values_col='N.Publicaciones',
                     x_label='Año',
                     linewidth=4,
                     markersize=10,
                     figsize_x=6,
                     figsize_y=4,
                     dynamic_ylim=True,
                     y_steps=50)
   #Transpose dataframe
   scopus_year_df = scopus_year_df.pivot_table(index=None,
                                                columns='Year',
                                                values='N.Publicaciones',
                                                fill_value=0).reset_index()
   scopus_year_df = util.calculate_percentage_growth_df(scopus_year_df, 2024, 2025)
   scopus_year_df = util.calcula_anual_percentage_growth_df(scopus_year_df, 2021, 2025)
   pub_data.append(settings.LINE)
   pub_data.append("Distribución de publicaciones en Scopus por año:")
   pub_data.append(scopus_year_df.to_string(index=False))

   #Process Web of Science data
   wos_df = loader.read_wos_data()
   #Filter wos by range
   mask = wos_df['Publication Year'].between(2021, 2025)
   range_wos_df = wos_df[mask]
   #Count publications by Publication Year
   wos_year_df = util.count_unique_data_by_column(range_wos_df,
                                                ['Publication Year'],
                                                'UT (Unique WOS ID)',
                                                'N.Publicaciones')
   ch.plot_lines_chart(wos_year_df.copy(),
                     x_col='Publication Year',
                     values_col='N.Publicaciones',
                     x_label='Año',
                     linewidth=4,
                     markersize=10,
                     figsize_x=6,
                     figsize_y=4,
                     dynamic_ylim=True,
                     y_steps=25)
   #Transpose dataframe
   wos_year_df = wos_year_df.pivot_table(index=None,
                                          columns='Publication Year',
                                          values='N.Publicaciones',
                                          fill_value=0).reset_index()
   wos_year_df = util.calculate_percentage_growth_df(wos_year_df, 2024, 2025)
   wos_year_df = util.calcula_anual_percentage_growth_df(wos_year_df, 2021, 2025)
   pub_data.append(settings.LINE)
   pub_data.append("Distribución de publicaciones en Web of Science por año:")
   pub_data.append(wos_year_df.to_string(index=False))

   #Analyze SJR and JCR data
   #####################################################################################
   #Get SJR and JCR data
   all_pub_df = loader.get_all_pure_publications_data()
   #Filter year_range between 2019 to 2025
   mask = all_pub_df['ANIO_PUBLICACION'].between(2021, 2025)
   all_pub_df = all_pub_df[mask]
   #Get all ids
   ids = all_pub_df['CODIGO_PURE'].unique().tolist()
   #Get SJR and JCR data
   sjr_jcr_df = loader.get_sjr_jcr_publications_data()
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
   pub_data.append(settings.LINE)
   #Plot line chart
   ch.plot_lines_chart_pivot(sjr_pivot_df.copy(),
                     category_col='SJR Best Quartile',
                     y_label='N. Publicaciones',
                     )
   sjr_growth_df = util.calculate_percentage_growth_df(sjr_pivot_df, 2024, 2025)
   sjr_growth_df = util.calcula_anual_percentage_growth_df(sjr_growth_df, 2021, 2025)
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
   pub_data.append(settings.LINE)
   #Plot line chart
   ch.plot_lines_chart_pivot(jcr_pivot_df.copy(),
                     category_col='JIF Quartile',
                     y_label='N. Publicaciones')
   jcr_growth_df = util.calculate_percentage_growth_df(jcr_pivot_df, 2024, 2025)
   jcr_growth_df = util.calcula_anual_percentage_growth_df(jcr_growth_df, 2021, 2025)
   pub_data.append(f"""Distribución de publicaciones por JCR y año de publicación y 
                     Crecimiento porcentual de publicaciones por JCR Quartile de 2024 a 2025:""")
   pub_data.append(jcr_growth_df.to_string(index=False))

   #Analyze citations data
   #####################################################################################
   #Get Scopus citations data
   citations_df = loader.get_scopus_citations_data()
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
   citations_year_df = util.calculate_percentage_growth_df(citations_year_df, 2024.0, 2025.0)
   citations_year_df = util.calcula_anual_percentage_growth_df(citations_year_df, 2021.0, 2025.0)
   pub_data.append(settings.LINE)
   pub_data.append("Distribución de citas en Scopus por año:")
   pub_data.append(citations_year_df.to_string(index=False))

   #Save data to txt file
   print('\n'.join(pub_data))
   util.save_txt_file('\n'.join(pub_data), f'output/publications_data_{year}.txt')
