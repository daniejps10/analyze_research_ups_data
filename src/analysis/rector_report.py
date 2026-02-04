import pandas as pd
from src.utils import util
from src import charts as ch
from src.data import loader
from src.config import settings

def generate_rector_book_graphs():
   book_data = []
   book_data.append(settings.LINE)
   #Method to generate graphs for rector's book
   print("Generating graphs for rector's book...")
   #Graphs of publications data
   #####################################################################################
   #Publications by year
   #--------------------------------------------------------------------------------------
   #Read all publications data
   all_pure_pub_df = loader.get_all_pure_publications_data()
   #Count publications by ANIO_PUBLICACION
   pub_year_df = util.count_unique_data_by_column(all_pure_pub_df,
                                                ['ANIO_PUBLICACION'],
                                                'CODIGO_PURE',
                                                'N.Publicaciones')
   #Filter years between 2018 and 2025
   mask = pub_year_df['ANIO_PUBLICACION'].between(2018, 2025)
   pub_year_df = pub_year_df[mask]
   #Plot line chart
   ch.plot_lines_chart(pub_year_df.copy(),
                     x_col='ANIO_PUBLICACION',
                     x_label='Año',
                     values_col='N.Publicaciones',
                     linewidth=4,
                     markersize=10,
                     figsize_x=6,
                     figsize_y=4,
                     dynamic_ylim=True,
                     y_steps=100)
   book_data.append("Publicaciones por año de publicación.")
   #Calculate percentage growth
   pub_year_growth_df = util.calculate_growth_cagr_acumulative_df(pub_year_df, 
                                                                  'ANIO_PUBLICACION', 
                                                                  'N.Publicaciones', 2021)
   book_data.append(pub_year_growth_df.to_string(index=False))

   #Count publications by year and tipo de producto
   count_pub_type_df = util.count_unique_data_by_column(all_pure_pub_df,
                                                ['ANIO_PUBLICACION','TIPO_PRODUCTO'],
                                                'CODIGO_PURE',
                                                'N.Publicaciones')
   #Filter years between 2018 and 2025
   mask = count_pub_type_df['ANIO_PUBLICACION'].between(2018, 2025)
   count_pub_type_df = count_pub_type_df[mask]
   PUB_TYPES = ['Artículo de revista',
               'Artículo de conferencia',
               'Capítulo de libro',
               'Libro']
   for pub_type in PUB_TYPES:
      type_pub_df = count_pub_type_df[count_pub_type_df['TIPO_PRODUCTO'].isin([pub_type])]
      #Drop TIPO_PRODUCTO column
      type_pub_df = type_pub_df.drop(columns=['TIPO_PRODUCTO']).drop_duplicates()

      #Plot line chart
      ch.plot_lines_chart(type_pub_df.copy(),
                        x_col='ANIO_PUBLICACION',
                        x_label='Año',
                        y_label='N. Publicaciones',
                        values_col='N.Publicaciones',
                        figsize_x=7,
                        figsize_y=4,
                        linewidth=4,
                        markersize=10,
                        dynamic_ylim=True,
                        fig_name=f'publicaciones_tipo_{pub_type}')
      #Calculate percentage growth
      type_pub_df = util.calculate_growth_cagr_acumulative_df(type_pub_df, 
                                                            'ANIO_PUBLICACION', 
                                                            'N.Publicaciones', 2021)
      book_data.append(settings.LINE)
      book_data.append(f"Publicaciones por año de publicación - Tipo de producto: {pub_type}.")
      book_data.append(type_pub_df.to_string(index=False))

   #Count publications by year and type
   pub_type_year_df = util.count_unique_data_by_column(all_pure_pub_df,
                                                ['ANIO_PUBLICACION', 'TIPO_PRODUCTO'],
                                                'CODIGO_PURE',
                                                'N.Publicaciones')
   #Filter years between 2018 and 2025
   mask = pub_type_year_df['ANIO_PUBLICACION'].between(2018, 2025)
   pub_type_year_df = pub_type_year_df[mask]
   book_data.append(settings.LINE)
   book_data.append("Publicaciones por tipo de producto y año de publicación.")
   book_data.append(pub_type_year_df.to_string(index=False))

   #Publications by CACES data type
   #--------------------------------------------------------------------------------------
   analysis_year = 2025
   filter_years = [analysis_year-1, analysis_year]
   #Get all pure publications from analysis years
   filter_pub_df = all_pure_pub_df[all_pure_pub_df['ANIO_PUBLICACION'].isin(filter_years)]
   filter_pub_ids = filter_pub_df['CODIGO_PURE'].unique().tolist()
   book_data.append(settings.LINE)
   book_data.append(f"Número de registros de publicaciones: {len(filter_pub_df)}")
   book_data.append(f"Publicaciones únicas -> Rango {filter_years}: {len(filter_pub_ids)}")
   #Count publications by year
   pub_year_df = util.count_unique_data_by_column(filter_pub_df,
                                                ['ANIO_PUBLICACION'],
                                                'CODIGO_PURE',
                                                'N.Publicaciones')
   book_data.append(f"Publicaciones por año del rango {filter_years}")
   book_data.append(pub_year_df.to_string(index=False))
   #Count publications by year and campus
   pub_campus_df = util.count_unique_data_by_column(filter_pub_df,
                                                ['ANIO_PUBLICACION', 'SEDE_ADMINISTRADOR'],
                                                'CODIGO_PURE',
                                                'N.Publicaciones')
   book_data.append(f"Publicaciones por sede del rango {filter_years}")
   book_data.append(pub_campus_df.to_string(index=False))

   #Get publication groups data
   pub_groups_df = loader.get_pub_groups_data()
   #Extract anio publicacion from all_pure_pub_df
   pub_groups_df = pub_groups_df.merge(all_pure_pub_df[['CODIGO_PURE', 
                                                      'ANIO_PUBLICACION', 
                                                      'TIPO_PRODUCTO']],
                                       on='CODIGO_PURE', how='left')
   filter_pub_groups_df = pub_groups_df[pub_groups_df['CODIGO_PURE'].isin(filter_pub_ids)].copy()
   filter_pub_groups_ids = filter_pub_groups_df['CODIGO_PURE'].unique().tolist()
   filter_pub_groups_df['ANIO_PUBLICACION'] = filter_pub_groups_df['ANIO_PUBLICACION'].astype(int)

   book_data.append(settings.LINE)
   book_data.append(f"Número de registros al expandir las publicaciones por grupos: {len(filter_pub_groups_df)}")
   book_data.append(f"Publicaciones únicas al expandir las publicaciones por grupos {filter_years}: {len(filter_pub_groups_ids)}")
   #Count publications by year
   pub_group_year_df = util.count_unique_data_by_column(filter_pub_groups_df,
                                                ['ANIO_PUBLICACION'],
                                                'CODIGO_PURE',
                                                'N.Publicaciones')
   book_data.append(f"Publicaciones por año del rango {filter_years}")
   book_data.append(pub_group_year_df.to_string(index=False))
   #Count publications by year and campus
   pub_group_year_campus_df = util.count_unique_data_by_column(filter_pub_groups_df,
                                                ['ANIO_PUBLICACION', 'SEDE_ADMINISTRADOR'],
                                                'CODIGO_PURE',
                                                'N.Publicaciones')
   book_data.append(f"Publicaciones únicas por sede del rango {filter_years}")
   book_data.append(pub_group_year_campus_df.to_string(index=False))
   #Count publications by year and type
   pub_group_year_type_df = util.count_unique_data_by_column(filter_pub_groups_df,
                                                ['ANIO_PUBLICACION', 'TIPO_PRODUCTO'],
                                                'CODIGO_PURE',
                                                'N.Publicaciones')
   book_data.append(f"Publicaciones únicas por tipo de producto del rango {filter_years}")
   book_data.append(pub_group_year_type_df.to_string(index=False))

   #Count publications by campus and year
   pub_year_campus_df = util.count_unique_data_by_column(filter_pub_groups_df,
                                                      ['SEDE_ADMINISTRADOR', 'ANIO_PUBLICACION'],
                                                      'CODIGO_PURE',
                                                      'N.Publicaciones')
   pub_year_campus_pivot_df = pub_year_campus_df.pivot(index='SEDE_ADMINISTRADOR', 
                                                      columns='ANIO_PUBLICACION', 
                                                      values='N.Publicaciones')
   pub_year_campus_pivot_df['Total'] = pub_year_campus_pivot_df.sum(axis=1)
   pub_year_campus_pivot_df = pub_year_campus_pivot_df.sort_values(by='Total', ascending=False).copy()
   pub_year_campus_pivot_df = pub_year_campus_pivot_df.drop(columns=['Total'])
   book_data.append(f"Publicaciones únicas por sede y año del rango {filter_years}")
   book_data.append(pub_year_campus_pivot_df.to_string())
   ch.plot_dynamic_bars(pub_year_campus_pivot_df.reset_index().copy(),
                           index_col='SEDE_ADMINISTRADOR',
                           y_label='N.Publicaciones',
                           figsize_x=6,
                           figsize_y=4,
                           fig_name=f'publicaciones_campus_year')


   #Count publications by year, campus and group legend
   pub_group_year_campus_legend_df = util.count_unique_data_by_column(filter_pub_groups_df,
                                                ['ANIO_PUBLICACION', 'SEDE_ADMINISTRADOR', 'GRUPO_LEYENDA'],
                                                'CODIGO_PURE',
                                                'N.Publicaciones')
   book_data.append(f"Publicaciones únicas por sede y grupo del rango {filter_years}")
   book_data.append(pub_group_year_campus_legend_df.to_string(index=False))
   CAMPUS_LIST = filter_pub_groups_df['SEDE_ADMINISTRADOR'].unique().tolist()
   for campus in CAMPUS_LIST:
      campus_groups_df = pub_group_year_campus_legend_df[pub_group_year_campus_legend_df['SEDE_ADMINISTRADOR'] == campus]
      book_data.append(f"Publicaciones únicas por grupo en la sede {campus} del rango {filter_years}")
      book_data.append(campus_groups_df.to_string(index=False))
      campus_pivot_df = campus_groups_df.pivot(index='GRUPO_LEYENDA', columns='ANIO_PUBLICACION', values='N.Publicaciones')
      #Sort data
      campus_pivot_df['Total'] = campus_pivot_df.sum(axis=1)
      campus_pivot_df = campus_pivot_df.sort_values(by='Total', ascending=False).copy()
      campus_pivot_df = campus_pivot_df.drop(columns=['Total'])
      ch.plot_stacked_bar_chart(campus_pivot_df.reset_index().copy(),
                              index_col='GRUPO_LEYENDA',
                              x_label='Grupo',
                              y_label='N.Publicaciones',
                              rotation_x_ticks=90,
                              figsize_x=12,
                              figsize_y=6,
                              custom_colors=True,
                              fig_name=f'pub_campus_{campus.replace(" ", "_").lower()}_groups')

   #Publications by type: CACES data type
   #--------------------------------------------------------------------------------------
   pure_caces_df = loader.get_pure_caces_data_type_df()
   CACES_COLS_DICT = {
      'SJR_Q1': 'SJR Q1',
      'SJR_Q2': 'SJR Q2',
      'SJR_Q3': 'SJR Q3',
      'SJR_Q4': 'SJR Q4',
      'CONF_Q1_Q2_Q3_Q4': 'Conf. Q1-Q4',
      'WOS_SCOPUS_NO_QUARTIL': 'Wos/Scopus NQ',
      'BASES_REGIONALES': 'Bases Regionales',
      'LATINDEX': 'Latindex',
      'CAP_LIBROS': 'Cap. Libros',
      'LIBROS': 'Libros',
      'OTROS_PUB': 'Otras pub.'
   }
   #Rename CACES columns
   pure_caces_df = pure_caces_df.rename(columns=CACES_COLS_DICT)
   CACES_COLS = list(CACES_COLS_DICT.values())
   KEEP_COLS = ['CODIGO_PURE'] + CACES_COLS
   #Keep relevant columns
   pure_caces_df = pure_caces_df[KEEP_COLS].drop_duplicates()
   #Merge with filter_pub_groups_df
   pub_groups_caces_df = filter_pub_groups_df.merge(pure_caces_df,
                                             on='CODIGO_PURE', how='left')
   #Publication year like int type
   pub_groups_caces_df['ANIO_PUBLICACION'] = pub_groups_caces_df['ANIO_PUBLICACION'].astype(int)
   pub_groups_caces_ids = pub_groups_caces_df['CODIGO_PURE'].unique().tolist()
   book_data.append(settings.LINE)
   book_data.append(f"Número de registros de publicaciones con CACES: {len(pub_groups_caces_df)}")
   book_data.append(f"Publicaciones únicas con CACES -> Rango {filter_years}: {len(pub_groups_caces_ids)}")
   
   #Count CACES data type by product type
   count_caces_type_df = util.count_unique_data_by_column(pub_groups_caces_df,
                                                ['ANIO_PUBLICACION', 'TIPO_PRODUCTO'],
                                                'CODIGO_PURE',
                                                'N.Publicaciones')
   book_data.append(f"Publicaciones por tipo de producto y tipo de CACES del rango {filter_years}")
   book_data.append(count_caces_type_df.to_string(index=False))

   #Count CACES data type by year
   count_caces_year_df = pub_groups_caces_df.groupby('ANIO_PUBLICACION')[CACES_COLS].nunique().reset_index()
   book_data.append(f"Publicaciones por tipo de CACES del rango {filter_years}")
   book_data.append(count_caces_year_df.to_string(index=False))
   #Plot stacked bar chart
   ch.plot_stacked_bar_chart(count_caces_year_df.copy(),
                           index_col='ANIO_PUBLICACION',
                           x_label='Año',
                           y_label='N.Publicaciones',
                           custom_colors=True,
                           fig_name='publications_ups')
   #Melt dataframe
   # melted_caces_df = count_caces_year_df.melt(id_vars=['ANIO_PUBLICACION'],
   #                                           value_vars=CACES_COLS,
   #                                           var_name='Categoría',
   #                                           value_name='N.Publicaciones')
   
   #Transpose dataframe
   t_caces_df = count_caces_year_df.set_index('ANIO_PUBLICACION').T.reset_index()
   t_caces_df = t_caces_df.rename(columns={'index': 'Categoría'})
   #Reset index
   book_data.append(f"Publicaciones por tipo de CACES del rango {filter_years} (transpuesta)")
   book_data.append(t_caces_df.to_string(index=False))
   #Plot stacked bar chart transposed
   ch.plot_stacked_bar_chart(t_caces_df.copy(),
                           index_col='Categoría',
                           x_label='Categoría',
                           y_label='N.Publicaciones',
                           rotation_x_ticks=90,
                           custom_colors=True,
                           fig_name='publications_ups_transposed')
   
   #Count CACES data type by year and campus, analysis year only
   analysis_pub_groups_caces_df = pub_groups_caces_df[pub_groups_caces_df['ANIO_PUBLICACION'] == analysis_year]
   count_caces_year_df = (analysis_pub_groups_caces_df.groupby(['SEDE_ADMINISTRADOR'])
                           [CACES_COLS].nunique().reset_index())
   #Transpose dataframe
   t_caces_campus_df = count_caces_year_df.set_index(['SEDE_ADMINISTRADOR']).T.reset_index()
   t_caces_campus_df = t_caces_campus_df.rename(columns={'index': 'Categoría'})
   book_data.append(f"Publicaciones por tipo de CACES y sede del año {analysis_year} (transpuesta)")
   book_data.append(t_caces_campus_df.to_string(index=False))
   #Plot grouped bar chart transposed
   ch.plot_stacked_bar_chart(t_caces_campus_df.copy(),
                           index_col='Categoría',
                           x_label='Categoría',
                           y_label='N.Publicaciones',
                           rotation_x_ticks=90,
                           figsize_x=10,
                           figsize_y=6,
                           custom_colors=True,
                           fig_name=f'analysis_{analysis_year}_publications_ups_campus_transposed')

   #Count CACES data type by year and campus
   count_caces_year_campus_df = (pub_groups_caces_df.groupby(['SEDE_ADMINISTRADOR', 'ANIO_PUBLICACION'])
                           [CACES_COLS].nunique().reset_index())
   CAMPUS_LIST = count_caces_year_campus_df['SEDE_ADMINISTRADOR'].unique().tolist()
   for campus in CAMPUS_LIST:
      campus_caces_df = count_caces_year_campus_df[count_caces_year_campus_df['SEDE_ADMINISTRADOR'] == campus]
      #Drop SEDE_ADMINISTRADOR column
      campus_caces_df = campus_caces_df.drop(columns=['SEDE_ADMINISTRADOR'])
      #Transpose dataframe
      t_caces_campus_df = campus_caces_df.set_index(['ANIO_PUBLICACION']).T.reset_index()
      t_caces_campus_df = t_caces_campus_df.rename(columns={'index': 'Categoría'})
      book_data.append(f"Publicaciones por tipo de CACES del campus {campus} (transpuesta)")
      book_data.append(t_caces_campus_df.to_string(index=False))
      #Plot stacked bar chart transposed
      ch.plot_stacked_bar_chart(t_caces_campus_df.copy(),
                              index_col='Categoría',
                              x_label='Categoría',
                              y_label='N.Publicaciones',
                              rotation_x_ticks=90,
                              figsize_x=10,
                              figsize_y=6,
                              custom_colors=True,
                              fig_name=f'publications_{campus.lower().replace(" ", "_")}_transposed')

   #Projects
   #--------------------------------------------------------------------------------------
   proy_df = loader.get_proy_data()

   #Filter by RANGO_INICIO between 2020 and 2025
   mask = proy_df['RANGO_INICIO'].between(2020, 2025)
   range_proy_df = proy_df[mask]

   #Count projects by init year
   count_proy_year_df = util.count_unique_data_by_column(range_proy_df,
                                                ['RANGO_INICIO'],
                                                'CODIGO_PROYECTO',
                                                'N.Proyectos')
   #Plot line chart
   ch.plot_lines_chart(count_proy_year_df.copy(),
                     x_col='RANGO_INICIO',
                     x_label='Año',
                     values_col='N.Proyectos',
                     linewidth=4,
                     markersize=10,
                     figsize_x=6,
                     figsize_y=4,
                     dynamic_ylim=True,
                     y_steps=20)
   count_proy_year_df = util.calculate_growth_cagr_acumulative_df(count_proy_year_df, 
                                                            'RANGO_INICIO', 
                                                            'N.Proyectos', 2021)
   book_data.append(settings.LINE)
   book_data.append("Proyectos por año de inicio.")
   book_data.append(count_proy_year_df.to_string(index=False))
   
   
   #Filter by RANGO_INICIO 2025
   mask = proy_df['RANGO_INICIO'].isin([2025])
   current_proy_df = proy_df[mask]
   #Count projects by area
   count_proy_area_df = util.count_unique_data_by_column(current_proy_df,
                                                ['AREA_CACES'],
                                                'CODIGO_PROYECTO',
                                                'N.Proyectos')
   book_data.append(settings.LINE)
   book_data.append("Proyectos por área CACES en el año 2025.")
   book_data.append(count_proy_area_df.to_string(index=False))

   #Save data to txt file
   print('\n'.join(book_data))
   util.save_txt_file('\n'.join(book_data), f'output/rector_book_graphs.txt')
