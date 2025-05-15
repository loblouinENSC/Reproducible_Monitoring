import pandas as pd


# SCRIPT INDEPENDANT (qui n'est pas executé dans l'app principale), 
# QUI PERMET DE GENERER UN FICHIER CSV QUI CONTIENT TOUT LES JOURS DE PANNES
# DEPUIS UN FICHIER CSV D'ENTREE QUI CONTIENT DES EVENEMENTS DE PANNES


INPUT_CSV_FILE = 'rule-door_failure_1week.csv'  # Nom de votre fichier d'entrée
OUTPUT_CSV_FILE = 'door_failure_days.csv'     # Nom du fichier de sortie qui sera généré

def generate_failure_days():
    """
    Lit les événements de panne de capteur à partir d'un fichier CSV,
    identifie les périodes de panne, et génère un nouveau fichier CSV
    listant tous les jours uniques durant lesquels le capteur était en panne.
    """
    try:
        # Lire le fichier CSV d'entrée.
        # Les colonnes sont supposées être : timestamp, annotation, status_code, et une valeur optionnelle.
        # status_code : 1 signifie que la panne commence, 0 signifie que la panne se termine.
        df = pd.read_csv(
            INPUT_CSV_FILE,
            delimiter=';',
            header=None,  # L'extrait ne montre pas de ligne d'en-tête
            names=['timestamp', 'annotation', 'status_code', 'value'], # Définition des noms de colonnes
            parse_dates=['timestamp'], # Interpréter la première colonne comme des dates/heures
            dtype={'annotation': str, 'status_code': int, 'value': str} # Spécifier les types de données
        )
    except FileNotFoundError:
        print(f"Erreur : Le fichier d'entrée '{INPUT_CSV_FILE}' n'a pas été trouvé.")
        return
    except Exception as e:
        print(f"Erreur lors de la lecture du fichier CSV '{INPUT_CSV_FILE}': {e}")
        return

    if df.empty:
        print(f"Le fichier d'entrée '{INPUT_CSV_FILE}' est vide. Aucun jour de panne à traiter.")
        # Créer un fichier de sortie vide avec en-tête
        pd.DataFrame(columns=['date']).to_csv(OUTPUT_CSV_FILE, index=False)
        print(f"Le fichier '{OUTPUT_CSV_FILE}' a été généré (vide).")
        return

    # Trier les événements par timestamp pour les traiter dans l'ordre chronologique
    df = df.sort_values(by='timestamp').reset_index(drop=True)

    all_failure_days = set()  # Utiliser un ensemble pour stocker les jours uniques de panne
    current_failure_start_time = None

    for index, row in df.iterrows():
        event_time = row['timestamp']
        status_code = row['status_code']

        if pd.isna(event_time): # Ignorer les lignes où le timestamp est manquant
            print(f"Avertissement : Timestamp manquant à la ligne {index + 2} (après tri). Ligne ignorée.")
            continue

        if status_code == 1:  # Début de la panne
            if current_failure_start_time is not None:
                # Un nouveau début de panne est signalé avant la fin explicite du précédent.
                # On considère que la panne précédente continue et que ce nouveau début est
                # soit redondant, soit indique un problème dans les données.
                # Pour la robustesse, on peut choisir de continuer avec le premier début non fermé
                # ou de prendre le dernier début comme le plus récent.
                # Ici, nous allons actualiser au nouveau début, en signalant l'ancien.
                print(f"Avertissement : Nouvelle période de panne commencée à {event_time.strftime('%Y-%m-%d %H:%M:%S')} "
                      f"alors qu'une panne précédente (démarrée à {current_failure_start_time.strftime('%Y-%m-%d %H:%M:%S')}) "
                      "n'était pas explicitement terminée. Le nouveau début est pris en compte.")
            current_failure_start_time = event_time
        
        elif status_code == 0:  # Fin de la panne
            if current_failure_start_time is not None:
                failure_end_time = event_time
                
                # Vérifier que l'heure de fin n'est pas antérieure à l'heure de début
                if failure_end_time < current_failure_start_time:
                    print(f"Avertissement : Fin de panne à {failure_end_time.strftime('%Y-%m-%d %H:%M:%S')} "
                          f"est antérieure au début de panne à {current_failure_start_time.strftime('%Y-%m-%d %H:%M:%S')}. "
                          "Cette période de panne est ignorée.")
                    current_failure_start_time = None # Réinitialiser, car cette période est invalide
                    continue

                # Normaliser les dates pour ne garder que le jour (heure mise à 00:00:00)
                start_date_norm = current_failure_start_time.normalize()
                end_date_norm = failure_end_time.normalize()
                
                # Générer toutes les dates uniques dans l'intervalle [start_date_norm, end_date_norm] inclus
                days_in_period = pd.date_range(start=start_date_norm, end=end_date_norm, freq='D')
                for day in days_in_period:
                    all_failure_days.add(day) # L'ajout à un ensemble gère automatiquement les doublons
                
                current_failure_start_time = None  # Réinitialiser pour la prochaine période de panne
            else:
                # Un événement de fin est rencontré sans qu'un début de panne ait été enregistré
                print(f"Information : Fin de panne à {event_time.strftime('%Y-%m-%d %H:%M:%S')} "
                      "rencontrée sans début de panne actif correspondant. Ignoré.")
    
    # Gérer le cas où une période de panne est toujours active à la fin du fichier
    # (c'est-à-dire, un statut '1' n'a pas été suivi d'un statut '0')
    if current_failure_start_time is not None:
        print(f"Information : La panne commencée à {current_failure_start_time.strftime('%Y-%m-%d %H:%M:%S')} "
              "est considérée comme toujours active à la fin du fichier source.")
        
        last_event_date_in_file = df['timestamp'].max() # Obtenir la date du dernier événement dans le fichier
        
        # S'assurer que last_event_date_in_file est valide et postérieur ou égal au début de la panne
        if pd.notna(last_event_date_in_file) and last_event_date_in_file >= current_failure_start_time:
            start_date_norm = current_failure_start_time.normalize()
            # La période de panne s'étend jusqu'au jour du dernier événement connu dans le fichier
            end_date_for_open_failure_norm = last_event_date_in_file.normalize()
            
            days_in_open_period = pd.date_range(start=start_date_norm, end=end_date_for_open_failure_norm, freq='D')
            for day in days_in_open_period:
                all_failure_days.add(day)
            print(f"Les jours de panne pour cette période ouverte non terminée ont été ajoutés "
                  f"jusqu'au {end_date_for_open_failure_norm.strftime('%Y-%m-%d')}.")

    if not all_failure_days:
        print("Aucun jour de panne n'a été identifié.")
        # Créer un fichier CSV vide avec seulement l'en-tête si aucun jour de panne n'est trouvé
        failure_output_df = pd.DataFrame(columns=['date'])
    else:
        # Convertir l'ensemble des jours de panne (qui sont des objets datetime) en une liste triée
        failure_dates_list_sorted = sorted(list(all_failure_days))
        # Créer un DataFrame Pandas à partir de la liste des dates
        failure_output_df = pd.DataFrame(failure_dates_list_sorted, columns=['date'])
        # Formater la colonne 'date' en chaînes de caractères YYYY-MM-DD pour le fichier CSV
        failure_output_df['date'] = failure_output_df['date'].dt.strftime('%Y-%m-%d')

    try:
        # Écrire le DataFrame dans le fichier CSV de sortie, sans l'index de Pandas
        failure_output_df.to_csv(OUTPUT_CSV_FILE, index=False)
        print(f"Le fichier '{OUTPUT_CSV_FILE}' a été généré avec succès, contenant {len(failure_output_df)} jours de panne.")
    except Exception as e:
        print(f"Erreur lors de l'écriture du fichier CSV '{OUTPUT_CSV_FILE}': {e}")

if __name__ == '__main__':
    generate_failure_days()