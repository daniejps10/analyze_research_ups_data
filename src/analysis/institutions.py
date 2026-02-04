from geopy.geocoders import Nominatim
import pycountry_convert as pc
from tqdm import tqdm
from src.data import loader
from src.config import settings

# Initialize geocoder (user_agent can be any name)
geolocator = Nominatim(user_agent="my_country_detector")

def get_region(name):
   try:
      # Search for the country (handles Spanish & English)
      location = geolocator.geocode(name, language='en', timeout=10)
      if not location:
         return "Not Found"

      # Extract country name and convert to 2-letter code (ISO)
      # We take the last part of the address which is usually the country
      full_address = location.address.split(",")
      country_en = full_address[-1].strip()
      
      country_code = pc.country_name_to_country_alpha2(country_en)
      continent_code = pc.country_alpha2_to_continent_code(country_code)

      # Map to your specific Spanish regions
      region_map = {
         'NA': 'América del Norte',
         'SA': 'América del Sur',
         'EU': 'Europa',
         'AF': 'África',
         'AS': 'Asia',
         'OC': 'Oceanía'
      }
      return region_map.get(continent_code, "Other")
   except Exception:
      return "Error"
   
def process_institutions_data(year: int):
   print(settings.LINE)
   print("Analyzing institutions data...")
   print(settings.LINE)

   #Process institutions data
   print('Get publications data...')
   pub_df = loader.get_all_pure_publications_data()
   #Apply filters
   pub_df = pub_df[pub_df['ANIO_PUBLICACION'].isin([year])]
   pub_ids = pub_df['CODIGO_PURE'].unique().tolist()
   print(f'Number of publications in {year}: {len(pub_ids)}')
   
   print('Get publication-author-organization data...')
   pub_aut_org_df = loader.get_all_pub_aut_org_data()
   #Extrac relevant columns
   pub_aut_org_df = pub_aut_org_df[['CODIGO_PURE', 'Source ID']]
   #Apply filters
   pub_aut_org_df = pub_aut_org_df[pub_aut_org_df['CODIGO_PURE'].isin(pub_ids)]
   #Drop na values
   pub_aut_org_df = pub_aut_org_df.dropna(subset=['Source ID']).drop_duplicates()
   #Keep only int Source ID values
   pub_aut_org_df = pub_aut_org_df[pub_aut_org_df['Source ID'].apply(lambda x: str(x).isdigit())]
   pub_aut_org_df['Source ID'] = pub_aut_org_df['Source ID'].astype(int)
   pub_aut_org_df = pub_aut_org_df.reset_index(drop=True).copy().drop_duplicates()
   ins_ids = pub_aut_org_df['Source ID'].unique().tolist()
   print(f'Number of publication-author-organization records with Scopus source ID in {year}: {len(pub_aut_org_df)}')

   #Read SciVal institutions data
   scival_inst_df = loader.get_sival_institutions_data()
   #Keep only relevant columns
   scival_inst_df = scival_inst_df[['Institution ID', 'Institution',
                                    'Sector', 'Country/Region']]
   scival_inst_df = scival_inst_df.drop_duplicates()
   scival_inst_df = scival_inst_df[scival_inst_df['Institution ID'].isin(ins_ids)]
   print(f'Number of SciVal institutions matching publication-author-organization records in {year}: {len(scival_inst_df)}')

   #Get region for each institution
   print('Getting regions for institutions...')
   countries_df = scival_inst_df[['Country/Region']].drop_duplicates().reset_index(drop=True)
   tqdm.pandas(desc="Mapping countries to regions")
   countries_df['Region'] = countries_df['Country/Region'].progress_apply(get_region)

   #Merge regions back to institutions
   scival_inst_df = scival_inst_df.merge(countries_df, on='Country/Region', how='left')

   #Save to Excel
   print('Saving to Excel...')
   scival_inst_df.to_excel(f'output/SciVal_Institutions_{year}.xlsx', index=False)
