import pandas as pd

# SCRIPT INDEPENDANT (qui n'est pas executé dans l'app principale), 
# QUI PERMET DE GENERER UN FICHIER CSV QUI CONTIENT TOUT LES JOURS DE PANNES
# DEPUIS UN FICHIER CSV D'ENTREE QUI CONTIENT DES EVENEMENTS DE PANNES

INPUT_CSV_FILE = 'rule-door_failure_1week.csv'  # Nom de votre fichier d'entrée
OUTPUT_CSV_FILE = 'door_failure_days.csv'      # Nom du fichier de sortie qui sera généré

# Définition de la date de fin de projet
PROJECT_END_DATE_STR = '14/12/17' # Date de fin de projet (JJ/MM/AA)

def generate_failure_days():
    """
    Lit les événements de panne de capteur à partir d'un fichier CSV,
    identifie les périodes de panne, et génère un nouveau fichier CSV
    listant tous les jours uniques durant lesquels le capteur était en panne.
    Si la dernière ligne du CSV d'entrée indique une panne active (status_code 1),
    la panne est considérée active jusqu'à la PROJECT_END_DATE.
    """
    try:
        project_end_date = pd.to_datetime(PROJECT_END_DATE_STR, format='%d/%m/%y').normalize()
    except ValueError as e:
        print(f"Erreur : La date de fin de projet '{PROJECT_END_DATE_STR}' est invalide. Veuillez utiliser le format JJ/MM/AA. Détails: {e}")
        return

    try:
        # Lire le fichier CSV d'entrée.
        df = pd.read_csv(
            INPUT_CSV_FILE,
            delimiter=';',
            header=None,
            names=['timestamp', 'annotation', 'status_code', 'value'],
            parse_dates=['timestamp'],
            dtype={'annotation': str, 'status_code': int, 'value': str}
        )
    except FileNotFoundError:
        print(f"Erreur : Le fichier d'entrée '{INPUT_CSV_FILE}' n'a pas été trouvé.")
        return
    except Exception as e:
        print(f"Erreur lors de la lecture du fichier CSV '{INPUT_CSV_FILE}': {e}")
        return

    if df.empty:
        print(f"Le fichier d'entrée '{INPUT_CSV_FILE}' est vide. Aucun jour de panne à traiter.")
        pd.DataFrame(columns=['date']).to_csv(OUTPUT_CSV_FILE, index=False)
        print(f"Le fichier '{OUTPUT_CSV_FILE}' a été généré (vide).")
        return

    df = df.sort_values(by='timestamp').reset_index(drop=True)

    all_failure_days = set()
    current_failure_start_time = None

    for index, row in df.iterrows():
        event_time = row['timestamp']
        status_code = row['status_code']

        if pd.isna(event_time):
            print(f"Avertissement : Timestamp manquant à la ligne {index + 2} (après tri). Ligne ignorée.")
            continue
        
        # Vérifier si status_code est NaN (peut arriver si la colonne n'est pas toujours un int valide)
        if pd.isna(status_code):
            print(f"Avertissement : status_code manquant ou invalide à la ligne {index + 2} (timestamp: {event_time}). Ligne ignorée.")
            continue

        try:
            # S'assurer que status_code est bien un entier pour la comparaison
            status_code = int(status_code)
        except ValueError:
            print(f"Avertissement : status_code '{status_code}' non entier à la ligne {index + 2} (timestamp: {event_time}). Ligne ignorée.")
            continue


        if status_code == 1:  # Début de la panne
            if current_failure_start_time is not None:
                print(f"Avertissement : Nouvelle période de panne commencée à {event_time.strftime('%Y-%m-%d %H:%M:%S')} "
                      f"alors qu'une panne précédente (démarrée à {current_failure_start_time.strftime('%Y-%m-%d %H:%M:%S')}) "
                      "n'était pas explicitement terminée. Le nouveau début est pris en compte.")
            current_failure_start_time = event_time
        
        elif status_code == 0:  # Fin de la panne
            if current_failure_start_time is not None:
                failure_end_time = event_time
                
                if failure_end_time < current_failure_start_time:
                    print(f"Avertissement : Fin de panne à {failure_end_time.strftime('%Y-%m-%d %H:%M:%S')} "
                          f"est antérieure au début de panne à {current_failure_start_time.strftime('%Y-%m-%d %H:%M:%S')}. "
                          "Cette période de panne est ignorée.")
                    current_failure_start_time = None
                    continue

                start_date_norm = current_failure_start_time.normalize()
                end_date_norm = failure_end_time.normalize()
                
                days_in_period = pd.date_range(start=start_date_norm, end=end_date_norm, freq='D')
                for day in days_in_period:
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
        # df ne peut pas être vide ici car current_failure_start_time est issu d'une ligne de df
        last_row = df.iloc[-1] 

        # Si la dernière ligne du fichier CSV (après tri) a un status_code de 1
        if last_row['status_code'] == 1 and current_failure_start_time == last_row['timestamp']:
            # La logique de la boucle principale assure que si la dernière ligne est un '1',
            # current_failure_start_time sera le timestamp de cette ligne.
            print(f"La dernière ligne du fichier CSV (à {last_row['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}, status_code=1) "
                  "indique que le capteur n'est jamais remis en route.")
            print(f"La panne est étendue jusqu'à la date de fin de projet : {project_end_date.strftime('%Y-%m-%d')}.")
            end_date_for_open_failure = project_end_date
        else:
            # La panne est ouverte, mais la dernière ligne du fichier n'était pas un '1' ayant initié cette panne active.
            # (ex: dernier '1' suivi de lignes invalides, ou 'current_failure_start_time' vient d'un '1' plus ancien
            # et la dernière ligne est un '0' mal placé ou invalide, ou un '1' dont le timestamp ne correspond pas - cas peu probable).
            # On utilise la date du dernier événement valide dans le fichier.
            print(f"La panne ouverte (commencée à {current_failure_start_time.strftime('%Y-%m-%d %H:%M:%S')}) "
                  "n'est pas terminée. La condition spéciale (dernière ligne avec status_code=1) n'est pas remplie "
                  "pour étendre jusqu'à la date de fin de projet.")
            
            last_valid_timestamp_in_file = df['timestamp'].max() # Timestamp du dernier événement valide
            
            if pd.notna(last_valid_timestamp_in_file) and last_valid_timestamp_in_file >= current_failure_start_time:
                print(f"Elle est étendue jusqu'au jour du dernier événement horodaté du fichier: "
                      f"{last_valid_timestamp_in_file.normalize().strftime('%Y-%m-%d')}.")
                end_date_for_open_failure = last_valid_timestamp_in_file.normalize()
            else:
                # Fallback si last_valid_timestamp_in_file est invalide ou antérieur
                print(f"Avertissement : Le dernier événement horodaté valide ({last_valid_timestamp_in_file}) "
                      f"n'est pas utilisable ou est antérieur à la panne ouverte ({current_failure_start_time}). "
                      "La panne sera étendue uniquement jusqu'à son jour de début.")
                end_date_for_open_failure = current_failure_start_time.normalize()
        
        # Ajouter les jours de panne pour la période ouverte, si une date de fin a été déterminée
        if end_date_for_open_failure is not None:
            if end_date_for_open_failure >= current_failure_start_time.normalize():
                start_date_norm = current_failure_start_time.normalize()
                # end_date_for_open_failure est déjà normalisée (soit par .normalize() soit project_end_date)
                
                days_in_open_period = pd.date_range(start=start_date_norm, end=end_date_for_open_failure, freq='D')
                for day in days_in_open_period:
                    all_failure_days.add(day)
                print(f"Les jours de panne pour cette période ouverte ont été ajoutés "
                      f"de {start_date_norm.strftime('%Y-%m-%d')} jusqu'au {end_date_for_open_failure.strftime('%Y-%m-%d')}.")
            else:
                # Cas où project_end_date ou last_valid_timestamp_in_file est antérieur à current_failure_start_time
                print(f"Avertissement : La date de fin calculée pour la panne ouverte ({end_date_for_open_failure.strftime('%Y-%m-%d')}) "
                      f"est antérieure à sa date de début ({current_failure_start_time.normalize().strftime('%Y-%m-%d')}). "
                      "Seul le jour de début ({current_failure_start_time.normalize().strftime('%Y-%m-%d')}) sera ajouté.")
                all_failure_days.add(current_failure_start_time.normalize())

    if not all_failure_days:
        print("Aucun jour de panne n'a été identifié.")
        failure_output_df = pd.DataFrame(columns=['date'])
    else:
        failure_dates_list_sorted = sorted(list(all_failure_days))
        failure_output_df = pd.DataFrame(failure_dates_list_sorted, columns=['date'])
        failure_output_df['date'] = failure_output_df['date'].dt.strftime('%Y-%m-%d')

    try:
        failure_output_df.to_csv(OUTPUT_CSV_FILE, index=False)
        print(f"Le fichier '{OUTPUT_CSV_FILE}' a été généré avec succès, contenant {len(failure_output_df)} jours de panne.")
    except Exception as e:
        print(f"Erreur lors de l'écriture du fichier CSV '{OUTPUT_CSV_FILE}': {e}")

if __name__ == '__main__':
    generate_failure_days()