import pandas as pd
import plotly.graph_objs as go
from dash import Dash, dcc, html, Input, Output 
import numpy as np

# --- Configuration ---
SLEEP_LOG_FILE = 'rule-sleep_quiet.csv'
BED_FAILURE_DAYS_FILE = 'sensors_failure_days/bed_failure_days.csv' 
APP_TITLE = "Sleeping Activity Dashboard"
TEXT_COLOR = 'white'
BACKGROUND_COLOR = '#111111'
DATA1_COLOR = '#36A0EB' #bleu
DATA2_COLOR = '#F14864' #rouge 
DATAMONTH_COLOR = '#36A0EB'

# --- Graph Configuration ---
LEGEND = dict(orientation="h",yanchor="bottom",y=1.1,xanchor="center",x=0.5)
MARGIN_CHART = dict(l=70, r=70, b=70, t=150, pad=3)
TITLE_X = 0.06
TITLE_Y = 0.92


# --- Data Loading and Processing Function --- 
def get_sleep_data():
    """Loads and preprocesses sleep and bed failure data."""
    try:
        sleep = pd.read_csv(SLEEP_LOG_FILE, delimiter=';', decimal=",",
                            names=["date", "annotation", "sleep_count", "duration"],
                            parse_dates=["date"], index_col="date")
        sleep['durationHr'] = sleep['duration'] / 3600
        sleep = sleep[['durationHr']]
    except FileNotFoundError:
        print(f"Error: '{SLEEP_LOG_FILE}' not found.")
        sleep = pd.DataFrame(columns=["durationHr"], index=pd.to_datetime([]))
        sleep.index.name = 'date'
    except Exception as e:
        print(f"Error loading {SLEEP_LOG_FILE}: {e}")
        sleep = pd.DataFrame(columns=["durationHr"], index=pd.to_datetime([]))
        sleep.index.name = 'date'

    # Initialisation des DataFrames pour les données d'échec
    bed_failure_daily_markers = pd.DataFrame(columns=['date'])
    bed_failure_source_for_monthly_agg = pd.DataFrame(columns=['failure_count'], index=pd.to_datetime([]))
    bed_failure_source_for_monthly_agg.index.name = 'date'

    try:
        failure_dates_df = pd.read_csv(
            BED_FAILURE_DAYS_FILE,
            header=None,         
            names=['date'],      
            parse_dates=[0],     
            comment='#'         
        )
        
        # bed_failure_daily_markers sera ce DataFrame directement, après nettoyage des dates non valides.
        bed_failure_daily_markers = failure_dates_df[['date']].dropna(subset=['date']).copy()
        # S'assurer que la colonne 'date' est bien de type datetime après la lecture et le dropna
        bed_failure_daily_markers['date'] = pd.to_datetime(bed_failure_daily_markers['date'], errors='coerce')
        bed_failure_daily_markers.dropna(subset=['date'], inplace=True)


        # Pour bed_failure_monthly_agg, nous créons un DataFrame source.
        # Chaque date dans bed_failure_daily_markers est un jour d'échec.
        if not bed_failure_daily_markers.empty:
            temp_bf_monthly = bed_failure_daily_markers.copy()
            temp_bf_monthly['failure_count'] = 1 # Chaque jour listé compte comme 1 jour d'échec
            bed_failure_source_for_monthly_agg = temp_bf_monthly.set_index('date')
        # else: bed_failure_source_for_monthly_agg reste vide (initialisé avant le try)

    except FileNotFoundError:
        print(f"Avertissement : Fichier des jours d'échec du lit '{BED_FAILURE_DAYS_FILE}' non trouvé.")
    except pd.errors.EmptyDataError: # Gérer le cas où le fichier est vide
        print(f"Avertissement : Fichier des jours d'échec du lit '{BED_FAILURE_DAYS_FILE}' est vide.")
    except Exception as e:
        print(f"Erreur lors du chargement du fichier des jours d'échec du lit '{BED_FAILURE_DAYS_FILE}': {e}")

    # ------Daily Aggregation------
    if not sleep.empty:
        sleep_daily = sleep.resample('D').agg(duration_sum=('durationHr', 'sum')).reset_index()
        sleep_daily['year_month'] = sleep_daily['date'].dt.to_period('M').astype(str)
    else:
        sleep_daily = pd.DataFrame(columns=['date', 'duration_sum', 'year_month'])
        sleep_daily = sleep_daily.astype({'date': 'datetime64[ns]', 'duration_sum': 'float64', 'year_month': 'object'})


    # -----Monthly Aggregation-----
    if not sleep_daily.empty:
        temp_daily_for_agg = sleep_daily[['date', 'duration_sum']].copy()
        temp_daily_for_agg['duration_sum_for_avg'] = temp_daily_for_agg['duration_sum'].replace(0, np.nan)
        sleep_monthly_agg = temp_daily_for_agg.set_index('date').resample('ME').agg(
            duration_mean=('duration_sum_for_avg', 'mean'),
            duration_sem=('duration_sum_for_avg', 'sem')
        )
    else:
        sleep_monthly_agg = pd.DataFrame(columns=['duration_mean', 'duration_sem'], index=pd.to_datetime([]))
        sleep_monthly_agg.index.name = 'date'

    # Bed Failure Monthly Aggregation
    if not bed_failure_source_for_monthly_agg.empty:
        bed_failure_monthly_agg = bed_failure_source_for_monthly_agg.resample('ME').agg(
            bed_failure_sum=('failure_count', 'sum') # Somme des jours d'échec dans le mois
        )
    else:
        bed_failure_monthly_agg = pd.DataFrame(columns=['bed_failure_sum'], index=pd.to_datetime([]))
        bed_failure_monthly_agg.index.name = 'date'


    # Merge and Reindex
    sleep_monthly = pd.merge(sleep_monthly_agg, bed_failure_monthly_agg, left_index=True, right_index=True, how='outer')
    if not sleep_monthly.empty:
        sleep_monthly.index = pd.to_datetime(sleep_monthly.index)
        start_date, end_date = sleep_monthly.index.min(), sleep_monthly.index.max()
        if pd.notna(start_date) and pd.notna(end_date):
            full_idx = pd.date_range(start=start_date, end=end_date, freq='ME')
            sleep_monthly = sleep_monthly.reindex(full_idx)
        
        sleep_monthly['duration_mean'] = sleep_monthly['duration_mean'].fillna(np.nan)
        sleep_monthly['duration_sem'] = np.where(sleep_monthly['duration_mean'].isna(), np.nan, sleep_monthly['duration_sem'].fillna(0))
        sleep_monthly['bed_failure_sum'] = sleep_monthly['bed_failure_sum'].fillna(0).astype(int)
    sleep_monthly = sleep_monthly.reset_index().rename(columns={'index': 'date'})

    if 'date' in sleep_monthly.columns and not sleep_monthly.empty:
        valid_dates_monthly = sleep_monthly['date'].notna()
        sleep_monthly['month_label'] = '' 
        if valid_dates_monthly.any():
             sleep_monthly.loc[valid_dates_monthly, 'month_label'] = sleep_monthly.loc[valid_dates_monthly, 'date'].dt.strftime('%m/%y')
    else: # Si sleep_monthly est vide ou n'a pas de colonne 'date'
        sleep_monthly['month_label'] = pd.Series(dtype='str')


    # Add year_month to bed_failure_daily_markers
    if not bed_failure_daily_markers.empty and 'date' in bed_failure_daily_markers.columns:
        if not bed_failure_daily_markers.empty: # Vérification supplémentaire après conversion potentielle en datetime
            bed_failure_daily_markers['year_month'] = bed_failure_daily_markers['date'].dt.to_period('M').astype(str)
        else:
            if 'year_month' not in bed_failure_daily_markers.columns: # Assurer que la colonne existe même si vide
                 bed_failure_daily_markers['year_month'] = pd.Series(dtype='str')
    else:
        if 'year_month' not in bed_failure_daily_markers.columns:
             bed_failure_daily_markers['year_month'] = pd.Series(dtype='str')


    return sleep_daily, sleep_monthly, bed_failure_daily_markers

# --- Figure Creation Function ---
def create_sleep_figure(daily_data, monthly_data, daily_failure_markers, scale, selected_month):
    """Creates the Plotly figure for sleep activity based on inputs."""
    fig = go.Figure()
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor=BACKGROUND_COLOR,
        plot_bgcolor=BACKGROUND_COLOR,
        font=dict(color=TEXT_COLOR),
        title=dict(font=dict(color=TEXT_COLOR)),
        legend=dict(font=dict(color=TEXT_COLOR)),
        xaxis=dict(title=dict(font=dict(color=TEXT_COLOR)), tickfont=dict(color=TEXT_COLOR), gridcolor='rgba(255, 255, 255, 0.1)'),
        yaxis=dict(title=dict(font=dict(color=TEXT_COLOR)), tickfont=dict(color=TEXT_COLOR), gridcolor='rgba(255, 255, 255, 0.1)'),
        yaxis2=dict(title=dict(font=dict(color=TEXT_COLOR)), tickfont=dict(color=TEXT_COLOR), gridcolor='rgba(255, 255, 255, 0.1)'),
        margin=MARGIN_CHART,
        hoverlabel=dict(
            font_size=16,             
            font_color="white",            
            namelength=-1 
        )
    )

    if scale == 'year':
        df_monthly = monthly_data
        if not df_monthly.empty:
            fig.add_trace(go.Bar(x=df_monthly['month_label'], y=df_monthly['duration_mean'], name="Durée moyenne sommeil (h)", error_y=dict(type='data', array=df_monthly['duration_sem']), marker_color=DATA1_COLOR))
            fig.add_trace(go.Scatter(x=df_monthly['month_label'], y=df_monthly['bed_failure_sum'], name="Jours échec lit", yaxis='y2', mode='lines+markers', line=dict(color=DATA2_COLOR, dash='dot')))
            
            y2_max_range = 5
            if pd.notna(df_monthly['bed_failure_sum'].max()) and df_monthly['bed_failure_sum'].max() > 0:
                y2_max_range = df_monthly['bed_failure_sum'].max() * 1.1

            fig.update_layout(
                title=dict(text="Vue Annuelle : Activité de Sommeil Mensuelle", x= TITLE_X, y = TITLE_Y),
                xaxis=dict(title=dict(text="Mois")),
                yaxis=dict(title=dict(text="Durée moyenne sommeil (h)"), tickfont=dict(color=DATA1_COLOR)),
                yaxis2=dict(title=dict(text="Jours échec lit"), overlaying='y', side='right', tickfont=dict(color=DATA2_COLOR), showgrid=False, range=[0, y2_max_range]),
                legend=LEGEND
            )
        else:
            fig.update_layout(title=dict(text="Yearly View: No monthly data available"))

    elif scale == 'month' and selected_month:
        df_daily_sleep = daily_data[daily_data['year_month'] == selected_month] if not daily_data.empty else pd.DataFrame()
        
        df_daily_failure_markers_filtered = pd.DataFrame() 
        if not daily_failure_markers.empty and 'year_month' in daily_failure_markers.columns and not daily_failure_markers[daily_failure_markers['year_month'] == selected_month].empty :
            df_daily_failure_markers_filtered = daily_failure_markers[daily_failure_markers['year_month'] == selected_month]

        # Déterminer les dates uniques pour l'axe X, en combinant sommeil et échecs
        all_relevant_dates = []
        if not df_daily_sleep.empty:
            all_relevant_dates.extend(df_daily_sleep['date'].tolist())
        if not df_daily_failure_markers_filtered.empty:
            all_relevant_dates.extend(df_daily_failure_markers_filtered['date'].tolist())
        
        unique_display_dates = []
        if all_relevant_dates:
            unique_display_dates = sorted(list(set(all_relevant_dates)))


        if not df_daily_sleep.empty:
            fig.add_trace(go.Bar(x=df_daily_sleep['date'], y=df_daily_sleep['duration_sum'], name="Durée sommeil (h)", marker_color=DATAMONTH_COLOR))
        
        if not df_daily_failure_markers_filtered.empty:
            fig.add_trace(go.Scatter(
                x=df_daily_failure_markers_filtered['date'], 
                y=[0.3] * len(df_daily_failure_markers_filtered), 
                name="Échec lit", 
                mode='markers', 
                marker=dict(color=DATA2_COLOR, size=10, symbol='x')
            ))
        
        if not df_daily_sleep.empty or not df_daily_failure_markers_filtered.empty:
             fig.update_layout(
                title=dict(text=f"Vue Journalière : {selected_month}", x= TITLE_X, y = TITLE_Y),
                xaxis=dict(title=dict(text="Jour"), 
                           tickformat='%d',
                           tickmode='array', 
                           tickvals=unique_display_dates, # Utiliser les dates uniques pour les ticks
                           ticktext=[d.strftime('%d') for d in unique_display_dates] if unique_display_dates else [],
                           dtick="D1" if not unique_display_dates else None # dtick peut être problématique avec tickvals explicites
                          ),
                yaxis=dict(title=dict(text="Durée sommeil (h)"), range=[0, 13]),
                legend=LEGEND
            )
        else: # Ni données de sommeil, ni échecs pour le mois sélectionné
            fig.update_layout(title=dict(text=f"Monthly View: No sleep data or bed failures for {selected_month}"))
            
    elif scale == 'month' and not selected_month:
        fig.update_layout(title=dict(text="Monthly View: Please select a month"))

    if not fig.data: 
        fig.update_layout(title=dict(text="Aucune donnée à afficher"))

    return fig


# --- Standalone App Execution  ---
if __name__ == '__main__':
    sleep_daily_data, sleep_monthly_data, sleep_bed_failure_daily_markers = get_sleep_data()
    
    available_months = []
    # Construire available_months à partir des données de sommeil ou d'échec
    temp_months_set = set()
    if not sleep_daily_data.empty and 'year_month' in sleep_daily_data.columns:
        temp_months_set.update(sleep_daily_data['year_month'].unique())
    if not sleep_bed_failure_daily_markers.empty and 'year_month' in sleep_bed_failure_daily_markers.columns:
        # Filtrer les NaN potentiels dans year_month avant d'appeler unique()
        valid_year_months = sleep_bed_failure_daily_markers['year_month'].dropna().unique()
        temp_months_set.update(valid_year_months)
    
    if temp_months_set:
        available_months = sorted(list(temp_months_set))

    app = Dash(__name__)
    app.title = APP_TITLE
    app.layout = html.Div(style={'backgroundColor': BACKGROUND_COLOR, 'color': TEXT_COLOR, 'padding': '20px'}, children=[
        html.H2(APP_TITLE, style={'textAlign': 'center', 'marginBottom': '30px'}),
        html.Div([
            html.Label("Select view scale:", style={'marginRight': '10px'}),
            dcc.RadioItems(
                id='scale-selector-sleep',
                options=[{'label': 'Year View (Monthly)', 'value': 'year'}, {'label': 'Month View (Daily)', 'value': 'month'}],
                value='year',
                labelStyle={'display': 'inline-block', 'marginRight': '20px'},
                inputStyle={'marginRight': '5px'}
            ),
        ], style={'marginBottom': '20px', 'textAlign': 'center'}),
        html.Div(id='month-dropdown-container-sleep', children=[
            html.Label("Select Month:", style={'marginRight': '10px'}),
            dcc.Dropdown(
                id='month-dropdown-sleep',
                options=[{'label': m, 'value': m} for m in available_months],
                value=available_months[0] if len(available_months) > 0 else None,
                clearable=False,
                style={'width': '200px', 'display': 'inline-block', 'color': '#333'}
            )
        ], style={'display': 'none', 'textAlign': 'center', 'marginBottom': '20px'}),
        dcc.Graph(id='sleep-activity-graph', style={'height': '60vh'})
    ])

    @app.callback(
        Output('month-dropdown-container-sleep', 'style'),
        Input('scale-selector-sleep', 'value')
    )
    def toggle_month_dropdown_sleep(scale):
        display = 'block' if scale == 'month' else 'none'
        return {'display': display, 'textAlign': 'center', 'marginBottom': '20px'}

    @app.callback(
        Output('sleep-activity-graph', 'figure'),
        Input('scale-selector-sleep', 'value'),
        Input('month-dropdown-sleep', 'value')
    )
    def update_graph_standalone(scale, selected_month):
        return create_sleep_figure(sleep_daily_data, sleep_monthly_data, sleep_bed_failure_daily_markers, scale, selected_month)

    app.run(debug=True)