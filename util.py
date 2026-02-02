import pandas as pd
import numpy as np
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

def calculate_percentage_growth(initial_value: float, end_value: float) -> float:
   #Apply formula (Vf-Vi)/Vi * 100
   if initial_value == 0:
      return 0.0
   return ((end_value - initial_value) / initial_value) * 100

def calculate_growth_cagr_acumulative_df(df: pd.DataFrame,
                                       year_col: str,
                                       values_col: str,
                                       anio_base: int = None) -> pd.DataFrame:
   # Sort values by year
   df = df.sort_values(by=year_col)

   # 1. Calculate the Growth %
   # pct_change() calculates (Current - Previous) / Previous
   growth_col = '% Crecimiento'
   df[growth_col] = df[values_col].pct_change() * 100
   df[growth_col] = df[growth_col].round(3)

   # 2. Calculate CAGR accumulative
   if anio_base is None:
      anio_base = df[year_col].iloc[0]
   value_base = df[df[year_col] == anio_base][values_col].values[0]

   # Función para calcular CAGR desde el inicio hasta la fila actual
   def calcular_cagr_fila(row):
      t = row[year_col] - anio_base
      if t == 0: return 0 # Para el primer año
      return (row[values_col] / value_base) ** (1 / t) - 1

   # Aplicar al dataframe
   cagr_col = f'CAGR_Acumulado_{anio_base}%'
   df[cagr_col] = df.apply(calcular_cagr_fila, axis=1) * 100
   df[cagr_col] = df[cagr_col].round(3)

   return df.drop_duplicates()

def calculate_percentage_growth_df(df: pd.DataFrame,
                                    col_init_values: str,
                                    col_end_values: str) -> pd.DataFrame:
   #Apply formula (Vf-Vi)/Vi * 100
   df["% Crecimiento"] = df.apply(
      lambda row: calculate_percentage_growth(row[col_init_values], row[col_end_values]), axis=1
   )
   return df.drop_duplicates()

def count_unique_data_by_column(df: pd.DataFrame, 
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

def calcula_anual_percentage_growth_df(df: pd.DataFrame,
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