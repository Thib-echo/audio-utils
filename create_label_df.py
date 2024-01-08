import pandas as pd
import multiprocessing as mp
from functools import partial
from unidecode import unidecode
from fuzzywuzzy import process, fuzz
import re
from tqdm import tqdm
import string
from pathlib import Path
from datetime import timedelta

def month_name_to_number(month_name):
    month_names = ['janvier', 'février', 'mars', 'avril', 'mai', 'juin', 
                   'juillet', 'août', 'septembre', 'octobre', 'novembre', 'décembre']
    try:
        return month_names.index(month_name.lower()) + 1
    except ValueError:
        return None
    
def extract_full_date(transcript):
    min_year = 1900

    # Pattern for numerical full date
    full_date_pattern = r'\b(0?[1-9]|[12][0-9]|3[01])[-/](0?[1-9]|1[0-2])[-/](\d{2}|\d{4})\b'
    
    # Pattern for full date with month name
    month_names = '|'.join(['janvier', 'février', 'mars', 'avril', 'mai', 'juin', 
                            'juillet', 'août', 'septembre', 'octobre', 'novembre', 'décembre'])
    date_with_month_pattern = r'\b(0?[1-9]|[12][0-9]|3[01])\s+(' + month_names + r')\s+(\d{4})\b'

    # Try to extract full date in numerical format
    full_dates = re.findall(full_date_pattern, transcript)
    formatted_dates = []
    if full_dates:
        for date in full_dates:
            day, month, year = date
            if len(year) == 2:  # Convert two-digit year to four digits
                year = "19" + year if year > "24" else "20" + year
            # Handle transcript erros (ex: 1568 instead of 1968)

            if int(year) < min_year:
                year = int("19" + year[-2:])

            formatted_dates.append(f"{day.zfill(2)}/{month.zfill(2)}/{str(year)}")
    
        return formatted_dates

    # Try to extract full date with month name
    dates_with_month = re.findall(date_with_month_pattern, transcript)
    if dates_with_month:
        for date in dates_with_month:
            day, month_name, year = date
            month_number = month_name_to_number(month_name)
            if month_number:
                formatted_dates.append(f"{day.zfill(2)}/{month_number:02d}/{year}")
        
        return formatted_dates

    # Extract just the year
    year_pattern = r'\b(19\d{2}|20\d{2})\b'
    years = re.findall(year_pattern, transcript)
    if years:
        year = int(years[0])
        if year < min_year:
            year = int("19" + year[-2:])
        return f"01/01/{str(year)}"  # Defaulting to '01/01/YYYY' for year-only cases

    return None

def extract_phone_numbers(transcript):
    phone_regex = r'(06|07)(\d{8})'

    transcript = transcript.replace(" ", "")
    transcript = transcript.translate(str.maketrans('', '', string.punctuation))
    match = re.search(phone_regex, transcript)
    match = match.group() if match else match
    return match

def extract_postcodes(transcript):
    # Existing pattern for postcodes like '75015' or '75-015'
    postcode_pattern = r'\b(75|77|78|91|92|93|94|95)(?:[-\s]?)(\d{3})\b'
    raw_postcodes = re.findall(postcode_pattern, transcript)

    # Additional pattern for postcodes written as '19ème'
    arrondissement_pattern = r'\b(\d{1,2})[eè]me\b'
    arrondissements = re.findall(arrondissement_pattern, transcript)

    # Formatting arrondissements to Paris postcodes (e.g., '75019' for '19ème')
    formatted_arrondissements = ['75' + arr.zfill(3) for arr in arrondissements]

    # Combining and returning all found postcodes
    return [''.join(match) for match in raw_postcodes] + formatted_arrondissements

def normalize_transcript(transcript):
    # Define a mapping of abbreviations to expansions
    abbreviations = {
        " dr ": " docteur ",
        " st ": " saint ",
        " ste ": " sainte ",
        # Add more abbreviations and their expansions as needed
    }
    # Ensure the address is a lowercase string
    normalized_transcript = unidecode(str(transcript).lower())
    # Replace abbreviations with their expanded form
    for abbr, expansion in abbreviations.items():
        normalized_transcript = normalized_transcript.replace(abbr, expansion)
    # Replace hyphens with spaces
    normalized_transcript = normalized_transcript.replace("-", " ")
    return normalized_transcript

def clean_trailing_words(address):
    stop_words = ['dans le', "c'est ca", " alors ", " a ", " et ", " avec ", '.', ',', '!', '?']

    for stop_word in stop_words:
        if stop_word in address:
            # Remove the stop word and anything that follows
            address = address.split(stop_word)[0]
            break
    return address.strip()

def extract_personal_info(data_folder):
    personal_info = {}

    # address_pattern = r'\b(\d+\s+(?:rue|rues|boulevard|avenue|place)\s+[a-zA-Zéèàêûôâîç\s-]+)\b'
    address_pattern = r'\b(\d+\s+(?:rue|rues|boulevard|avenue|place|square|villa|quai|allée|chaussée|passage)\s+[a-zA-Zéèàêûôâîç\s-]+(?:\'[a-zA-Zéèàêûôâîç\s-]+)?)\b'

    # address_pattern = r'\b\d+\s*(rue|rues|avenue|boulevard|place)\s*[^\d,]+\b'
    number_sequence_pattern = r'\b\d+\b'

    for file_path in tqdm(data_folder.rglob('*.txt')):
        with open(file_path, 'r', encoding='utf-8') as file:
            transcript = file.read()
            transcript = normalize_transcript(transcript)

            personal_info[str(Path(file_path.name))] = {
                'date_of_birth': {
                    'full_date': extract_full_date(transcript)
                },
                'addresses': list(set([clean_trailing_words(address) for address in re.findall(address_pattern, transcript)])),
                'postcodes': extract_postcodes(transcript),
                'phone_numbers': extract_phone_numbers(transcript),
                # 'names' : list(set(find_names_with_spacy(transcript))),
                # 'other_findings': re.findall(number_sequence_pattern, transcript),
            }
        
    return personal_info


def extract_road_number(address):
    match = re.search(r'\b\d+\b', address)
    return match.group(0) if match else None

def road_numbers_match(csv_road_number, input_road_number):
    # If the CSV address has 'nan' for the road number, we treat it as a match
    if csv_road_number == 'nan':
        return True
    # Check if the input road number is contained within the CSV road number or vice versa
    return input_road_number in csv_road_number or csv_road_number in input_road_number

# Define your core matching logic here (similar to your original find_matching_row function)
def find_matching_row_partial(csv_path, data_dict):
    # Define the columns to read
    cols_to_use = ['Date', 'Annee', 'Mois', 'Jour', 'Heure', 'Min', 'FullAdresse', 'TelPatient', 'DateNaiss', 'CodePostal', 'Nom', 'Prenom', 'Devenir', 'NumRue']

    # Read the CSV just once
    df = pd.read_csv(csv_path, usecols=cols_to_use, sep=',', encoding='utf-8')
    df.dropna(subset=['Devenir'], inplace=True)

    df.fillna({'Annee': 0, 'Mois': 0, 'Jour': 0, 'Heure': 0, 'Min': 0, 'Sec': 0}, inplace=True)
    df['DateTime'] = pd.to_datetime(df['Annee'].astype(int).astype(str) + '-' +
                                    df['Mois'].astype(int).astype(str).str.zfill(2) + '-' +
                                    df['Jour'].astype(int).astype(str).str.zfill(2) + ' ' +
                                    df['Heure'].astype(int).astype(str).str.zfill(2) + ':' +
                                    df['Min'].astype(int).astype(str).str.zfill(2),
                                    errors='coerce')
    df['DateNaiss'] = pd.to_datetime(df['DateNaiss'], format='%Y-%m-%d', errors='coerce')

    # Preprocess 'FullAdresse'
    df['FullAdresse'] = df['FullAdresse'].apply(lambda x: unidecode(str(x).lower()).replace("-", " "))

    # Prepare results dictionary
    results = {}

    for file_path, info in tqdm(data_dict.items()):
        # Extract and convert full datetime from file_path
        datetime_parts = file_path.split('_')[:6]
        datetime_str = '-'.join(datetime_parts[:3]) + ' ' + ':'.join(datetime_parts[3:6])
        file_datetime = pd.to_datetime(datetime_str)

        # Calculate 4-hour window
        start_time = file_datetime - timedelta(hours=2)
        end_time = file_datetime + timedelta(hours=2)

        # Filter the dataframe by the 4-hour window
        date_filtered_df = df[(df['DateTime'] >= start_time) & (df['DateTime'] <= end_time)]

        if info.get('phone_numbers'):
            transcript_phone_number = info['phone_numbers']
            medical_db_phone_number = date_filtered_df['TelPatient'].astype('str').str.contains(transcript_phone_number, na=False)
            match = date_filtered_df[medical_db_phone_number]
            if not match.empty:
                results[file_path] = (match, "phone_numbers")
                continue

        if info.get('addresses'):
            for address in info['addresses']:
                list_adresse = date_filtered_df['FullAdresse'].tolist()
                best_match = process.extractOne(address, list_adresse, scorer=fuzz.token_sort_ratio)
                if best_match:

                    # If we have a perfect match, we're done
                    if best_match[1] == 100:
                        match = date_filtered_df[date_filtered_df['FullAdresse'].str.contains(re.escape(best_match[0]), na=False)]
                        if not match.empty:
                            results[file_path] = (match, "addresses")
                            break

                    # For high-scoring matches, we check the road numbers
                    elif best_match[1] > 80:
                        input_road_number = extract_road_number(address)
                        csv_road_number = extract_road_number(best_match[0]) or 'nan'  # Treat missing numbers as 'nan'
                        if road_numbers_match(csv_road_number, input_road_number):
                            match = date_filtered_df[date_filtered_df['FullAdresse'].str.contains(re.escape(best_match[0]), na=False)]
                            if not match.empty:
                                results[file_path] = (match, "addresses")
                                break

        if info.get('date_of_birth', {}).get('full_date'):
            for dob in info['date_of_birth']['full_date']:
                dob = pd.to_datetime(dob, dayfirst=True, errors='coerce')
                if dob is not pd.NaT:
                    match = date_filtered_df[date_filtered_df['DateNaiss'] == dob]
                    if not match.empty:
                        results[file_path] = (match, "date_of_birth")
                        break

        if info.get('postcodes'):
            for postcode in info['postcodes']:
                match = date_filtered_df[date_filtered_df['CodePostal'] == str(postcode)]
                if not match.empty:
                    results[file_path] = (match, "postcodes")
                    break

    return results

# Function to handle multiprocessing
def find_matching_row_multiprocessing(csv_path, data_dict, num_processes=None):
    num_processes = num_processes or mp.cpu_count()
    items = list(data_dict.items())
    chunk_size = len(items) // num_processes
    data_chunks = [dict(items[i:i + chunk_size]) for i in range(0, len(items), chunk_size)]
    pool = mp.Pool(processes=num_processes)
    func = partial(find_matching_row_partial, csv_path)
    results = pool.map(func, data_chunks)
    pool.close()
    pool.join()
    combined_results = {}
    for result in results:
        combined_results.update(result)
    return combined_results

# Function to create a label DataFrame
def create_label_dataframe(datafolder, matches):
    summary_data = []
    for file_path, (match_df, _) in matches.items():
        transcript_path = Path.joinpath(datafolder, '_'.join(file_path.split('_')[:-1]), file_path)
        with open(transcript_path, 'r', encoding='utf-8') as file:
            transcript = file.read()
            row = match_df.iloc[0]
            devenir_value = int(row['Devenir'])
            summary_data.append({'file_path': file_path, 'transcript': transcript, 'Devenir': devenir_value})
    summary_df = pd.DataFrame(summary_data)
    return summary_df

# Example usage
if __name__ == '__main__':
    medical_db_path = "../audio_database/base_medicale//BaseMed2022LR_cleaned.csv"
    datafolder = Path("../audio_database/big_db_test/")
    personnal_data = extract_personal_info(datafolder)
    num_processes = 8  # Adjust based on your system capabilities

    # Find matching rows using multiprocessing
    matches = find_matching_row_multiprocessing(medical_db_path, personnal_data, num_processes)

    # Create the label DataFrame
    summary_df = create_label_dataframe(datafolder, matches)

    # Display the first few rows of the DataFrame
    print(summary_df.head())

    # Save the DataFrame to a CSV file
    summary_df.to_csv("output_labels.csv", index=False)
