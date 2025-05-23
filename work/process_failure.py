import pandas as pd
import argparse
import os

# SCRIPT EXECUTE AU MOMENT DU MAKE,
# QUI PERMET DE GENERER UN FICHIER CSV QUI CONTIENT TOUT LES JOURS DE PANNES 
# (pas seulement le jour du début de la panne, mais tous les jours compris dans la période de panne)
# DEPUIS UN FICHIER CSV D'ENTREE QUI CONTIENT DES EVENEMENTS DE PANNES

parser = argparse.ArgumentParser(description="Générer les jours de pannes pour un participant.")
parser.add_argument("participant_number", type=int, help="Numéro du participant (ex: 1)")

args = parser.parse_args()
PARTICIPANT_NUMBER = args.participant_number

print(f"Participant numéro : {PARTICIPANT_NUMBER}")

# Définition de la période du projet
PROJECT_START_DATE_STR = '01/01/17' # Date de début de projet (JJ/MM/AA)
PROJECT_END_DATE_STR = '14/12/17' # Date de fin de projet (JJ/MM/AA)

def generate_failure_days_for_file(input_csv_file, output_csv_file, project_start_date, project_end_date):
    """
    Lit les événements de panne de capteur à partir d'un fichier CSV donné,
    identifie les périodes de panne, et génère un nouveau fichier CSV
    listant tous les jours uniques durant lesquels le capteur était en panne,
    uniquement à partir de la date de début de projet.
    Si la dernière ligne du CSV d'entrée indique une panne active (status_code 1),
    la panne est considérée active jusqu'à la PROJECT_END_DATE.
    """
    print(f"\n--- Traitement de {input_csv_file} ---")

    # Assure que le répertoire de sortie existe
    output_dir = os.path.dirname(output_csv_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Répertoire de sortie créé : {output_dir}")

    try:
        # Lire le fichier CSV d'entrée.
        df = pd.read_csv(
            input_csv_file,
            delimiter=';',
            header=None,
            names=['timestamp', 'annotation', 'status_code', 'value'],
            parse_dates=['timestamp'],
            dtype={'annotation': str, 'status_code': int, 'value': str}
        )
    except FileNotFoundError:
        print(f"Erreur : Le fichier d'entrée '{input_csv_file}' n'a pas été trouvé. Ignoré.")
        return
    except Exception as e:
        print(f"Erreur lors de la lecture du fichier CSV '{input_csv_file}': {e}. Ignoré.")
        return

    if df.empty:
        print(f"Le fichier d'entrée '{input_csv_file}' est vide. Aucun jour de panne à traiter.")
        pd.DataFrame(columns=['date']).to_csv(output_csv_file, index=False)
        print(f"Le fichier '{output_csv_file}' a été généré (vide).")
        return

    # Filtrer les événements pour ne garder que ceux qui sont à partir de la date de début de projet
    df = df[df['timestamp'].notna() & (df['timestamp'] >= project_start_date)].copy()
    df = df.sort_values(by='timestamp').reset_index(drop=True)

    if df.empty:
        print(f"Après filtrage par la date de début de projet ({project_start_date.strftime('%d/%m/%y')}), le fichier '{input_csv_file}' est vide. Aucun jour de panne à traiter.")
        pd.DataFrame(columns=['date']).to_csv(output_csv_file, index=False)
        print(f"Le fichier '{output_csv_file}' a été généré (vide).")
        return

    all_failure_days = set()
    current_failure_start_time = None

    for index, row in df.iterrows():
        event_time = row['timestamp']
        status_code = row['status_code']

        if pd.isna(event_time):
            print(f"Avertissement : Timestamp manquant à la ligne {index + 2} (après tri et filtrage) dans '{input_csv_file}'. Ligne ignorée.")
            continue
        
        if pd.isna(status_code):
            print(f"Avertissement : status_code manquant ou invalide à la ligne {index + 2} (timestamp: {event_time}) dans '{input_csv_file}'. Ligne ignorée.")
            continue

        try:
            status_code = int(status_code)
        except ValueError:
            print(f"Avertissement : status_code '{status_code}' non entier à la ligne {index + 2} (timestamp: {event_time}) dans '{input_csv_file}'. Ligne ignorée.")
            continue

        if status_code == 1:  # Début de la panne
            potential_failure_start = max(event_time, project_start_date)
            if current_failure_start_time is not None:
                print(f"Avertissement : Nouvelle période de panne commencée à {event_time.strftime('%Y-%m-%d %H:%M:%S')} "
                      f"alors qu'une panne précédente (démarrée à {current_failure_start_time.strftime('%Y-%m-%d %H:%M:%S')}) "
                      "n'était pas explicitement terminée. Le nouveau début est pris en compte.")
            current_failure_start_time = potential_failure_start
        
        elif status_code == 0:  # Fin de la panne
            if current_failure_start_time is not None:
                failure_end_time = event_time
                
                effective_start_time = max(current_failure_start_time, project_start_date)

                if failure_end_time < effective_start_time:
                    print(f"Avertissement : Fin de panne à {failure_end_time.strftime('%Y-%m-%d %H:%M:%S')} "
                          f"est antérieure au début de panne ajusté à {effective_start_time.strftime('%Y-%m-%d %H:%M:%S')}. "
                          "Cette période de panne est ignorée.")
                    current_failure_start_time = None
                    continue

                start_date_norm = effective_start_time.normalize()
                end_date_norm = failure_end_time.normalize()
                
                end_date_norm = min(end_date_norm, project_end_date)

                if start_date_norm <= end_date_norm:
                    days_in_period = pd.date_range(start=start_date_norm, end=end_date_norm, freq='D')
                    for day in days_in_period:
                        if project_start_date.normalize() <= day <= project_end_date.normalize():
                            all_failure_days.add(day)
                
                current_failure_start_time = None
            else:
                print(f"Information : Fin de panne à {event_time.strftime('%Y-%m-%d %H:%M:%S')} "
                      "rencontrée sans début de panne actif correspondant. Ignoré.")
    
    # Gérer le cas où une période de panne est toujours active à la fin du fichier
    if current_failure_start_time is not None:
        print(f"Information : La panne commencée à {current_failure_start_time.strftime('%Y-%m-%d %H:%M:%S')} "
              "est considérée comme active à la fin du traitement des événements du fichier source.")

        end_date_for_open_failure = None
        last_row_after_filter = df.iloc[-1] 

        if last_row_after_filter['status_code'] == 1 and current_failure_start_time.normalize() == max(last_row_after_filter['timestamp'], project_start_date).normalize():
            print(f"La dernière ligne du fichier CSV (à {last_row_after_filter['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}, status_code=1) "
                  "indique que le capteur n'est jamais remis en route.")
            print(f"La panne est étendue jusqu'à la date de fin de projet : {project_end_date.strftime('%Y-%m-%d')}.")
            end_date_for_open_failure = project_end_date
        else:
            last_valid_timestamp_in_project_range = df['timestamp'].max()
            
            if pd.notna(last_valid_timestamp_in_project_range) and last_valid_timestamp_in_project_range >= current_failure_start_time:
                print(f"Elle est étendue jusqu'au jour du dernier événement horodaté du fichier (dans la plage du projet): "
                      f"{last_valid_timestamp_in_project_range.normalize().strftime('%Y-%m-%d')}.")
                end_date_for_open_failure = last_valid_timestamp_in_project_range.normalize()
            else:
                print(f"Avertissement : Le dernier événement horodaté valide ({last_valid_timestamp_in_project_range}) "
                      f"n'est pas utilisable ou est antérieur à la panne ouverte ({current_failure_start_time}). "
                      "La panne sera étendue uniquement jusqu'à son jour de début ajusté.")
                end_date_for_open_failure = current_failure_start_time.normalize()
        
        if end_date_for_open_failure is not None:
            start_date_for_open_period = max(current_failure_start_time.normalize(), project_start_date.normalize())

            if end_date_for_open_failure >= start_date_for_open_period:
                days_in_open_period = pd.date_range(start=start_date_for_open_period, end=end_date_for_open_failure, freq='D')
                for day in days_in_open_period:
                    if project_start_date.normalize() <= day <= project_end_date.normalize():
                        all_failure_days.add(day)
                print(f"Les jours de panne pour cette période ouverte ont été ajoutés "
                      f"de {start_date_for_open_period.strftime('%Y-%m-%d')} jusqu'au {end_date_for_open_failure.strftime('%Y-%m-%d')}.")
            else:
                print(f"Avertissement : La date de fin calculée pour la panne ouverte ({end_date_for_open_failure.strftime('%Y-%m-%d')}) "
                      f"est antérieure à sa date de début ajustée ({start_date_for_open_period.strftime('%Y-%m-%d')}). "
                      "Aucun jour supplémentaire ajouté pour cette panne ouverte.")

    if not all_failure_days:
        print(f"Aucun jour de panne n'a été identifié dans la période du projet pour '{input_csv_file}'.")
        failure_output_df = pd.DataFrame(columns=['date'])
    else:
        filtered_failure_days = {d for d in all_failure_days if project_start_date.normalize() <= d <= project_end_date.normalize()}
        failure_dates_list_sorted = sorted(list(filtered_failure_days))
        failure_output_df = pd.DataFrame(failure_dates_list_sorted, columns=['date'])
        failure_output_df['date'] = failure_output_df['date'].dt.strftime('%Y-%m-%d')

    try:
        failure_output_df.to_csv(output_csv_file, index=False)
        print(f"Le fichier '{output_csv_file}' a été généré avec succès, contenant {len(failure_output_df)} jours de panne.")
    except Exception as e:
        print(f"Erreur lors de l'écriture du fichier CSV '{output_csv_file}': {e}")

if __name__ == '__main__':
    # Liste des paires fichier d'entrée / fichier de sortie
    #Si nouveaux fichiers de règles sont ajoutés, il faut les ajouter ici : 
    files_to_process = [
        (f'work/participant_{PARTICIPANT_NUMBER}/rules/rule-door_failure_1week.csv', f'work/participant_{PARTICIPANT_NUMBER}/sensors_failure_days/door_failure_days.csv'),
        (f'work/participant_{PARTICIPANT_NUMBER}/rules/rule-bed_failure.csv', f'work/participant_{PARTICIPANT_NUMBER}/sensors_failure_days/bed_failure_days.csv'),
        (f'work/participant_{PARTICIPANT_NUMBER}/rules/rule-platform_failure_1day.csv', f'work/participant_{PARTICIPANT_NUMBER}/sensors_failure_days/platform_failure_days.csv'),
        (f'work/participant_{PARTICIPANT_NUMBER}/rules/rule-toilet_failure.csv', f'work/participant_{PARTICIPANT_NUMBER}/sensors_failure_days/toilet_failure_days.csv'),
    ]

    try:
        project_start_date = pd.to_datetime(PROJECT_START_DATE_STR, format='%d/%m/%y').normalize()
        project_end_date = pd.to_datetime(PROJECT_END_DATE_STR, format='%d/%m/%y').normalize()
    except ValueError as e:
        print(f"Erreur fatale : Une des dates de projet est invalide. Veuillez utiliser le format JJ/MM/AA. Détails: {e}")
        exit(1) # Quitte le script si les dates de projet sont invalides

    if project_start_date > project_end_date:
        print("Erreur fatale : La date de début de projet ne peut pas être postérieure à la date de fin de projet.")
        exit(1) # Quitte le script si les dates de projet sont invalides

    for input_file, output_file in files_to_process:
        generate_failure_days_for_file(input_file, output_file, project_start_date, project_end_date)