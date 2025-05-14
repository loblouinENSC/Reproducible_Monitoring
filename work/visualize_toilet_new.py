import pandas as pd
import plotly.graph_objs as go
from dash import Dash, dcc, html, Input, Output
import numpy as np # Import numpy for potential use

# --- Configuration ---
TOILET_LOG_FILE = 'rule-toilet.csv'
APP_TITLE = "Toilet Activity Viewer"
TEXT_COLOR = 'white'
BACKGROUND_COLOR = '#111111'
DATA1_COLOR = '#36A0EB' 
DATA2_COLOR = '#43D37B' # green
DATA3_COLOR = '#EB9636'
DATAMONTH_COLOR = DATA1_COLOR
DATAMONTH2_COLOR = DATA2_COLOR

# --- Graph Configuration ---
LEGEND = dict(orientation="h",yanchor="bottom",y=1.1,xanchor="center",x=0.5) 
MARGIN_CHART = dict(l=70, r=70, b=70, t=150, pad=3) 

TITLE_X = 0.06
TITLE_Y = 0.92



# --- Data Loading and Processing Function --- #
def get_toilet_data():
    """Loads and preprocesses toilet activity data, returning daily and monthly aggregations."""
    try:
        activity = pd.read_csv(TOILET_LOG_FILE, delimiter=';', decimal=",",
                                names=["date", "annotation", "activity_count", "duration"],
                                parse_dates=["date"], index_col="date")
        # Conversion de la durée de secondes en minutes
        activity['duration_min'] = activity['duration'] / 60
    except FileNotFoundError:
        print(f"Error: '{TOILET_LOG_FILE}' not found.")
        activity = pd.DataFrame(columns=["annotation", "activity_count", "duration","duration_min"],
                                 index=pd.to_datetime([]))
        activity.index.name = 'date'
    except Exception as e:
        print(f"Error loading {TOILET_LOG_FILE}: {e}")
        activity = pd.DataFrame(columns=["annotation", "activity_count", "duration","duration_min"],
                                 index=pd.to_datetime([]))
        activity.index.name = 'date'

    # Agrégation Journalière intermédiaire pour les calculs mensuels
    # Calcule la somme des durées et des passages par jour.
    if not activity.empty:
        activity_daily_intermediate = activity.resample('D').agg(
            activity_count_sum_daily=('activity_count', 'sum'), # Somme des passages pour ce jour
            duration_min_sum_daily=('duration_min', 'sum')    # Somme totale des durées pour ce jour
        )
    else:
        activity_daily_intermediate = pd.DataFrame(
            columns=['activity_count_sum_daily', 'duration_min_sum_daily'],
            index=pd.to_datetime([])
        )
        activity_daily_intermediate.index.name = 'date'

    # Préparation de 'activity_daily' pour la vue journalière du graphique
    # Le graphique journalier a besoin de : date, duration_min_sum, activity_count_sum, year_month
    if not activity.empty:
         activity_daily_for_graph = activity.resample('D').agg(
            activity_count_sum=('activity_count', 'sum'), # Total des passages ce jour-là
            duration_min_sum=('duration_min', 'sum')      # Total des minutes ce jour-là
        ).reset_index()
         activity_daily_for_graph['year_month'] = activity_daily_for_graph['date'].dt.to_period('M').astype(str)
    else:
        activity_daily_for_graph = pd.DataFrame(columns=['date', 'activity_count_sum', 'duration_min_sum', 'year_month'])


    # Monthly Aggregation 
    if not activity_daily_intermediate.empty:
        # Préparer les totaux quotidiens (duration_min_sum_daily) pour la moyenne mensuelle.
        # Remplacer 0 par NaN pour que .mean() et .sem() les ignorent,
        # ainsi les jours sans aucune activité ne sont pas comptés comme "0 minute".
        monthly_agg_input = activity_daily_intermediate.copy()
        monthly_agg_input['duration_min_sum_daily_for_avg'] = monthly_agg_input['duration_min_sum_daily'].replace(0, np.nan)

        activity_monthly = monthly_agg_input.resample('ME').agg(
            # Moyenne mensuelle des SOMMES QUOTIDIENNES de durée
            duration_min_mean_of_daily_totals=('duration_min_sum_daily_for_avg', 'mean'),
            duration_min_sem_of_daily_totals=('duration_min_sum_daily_for_avg', 'sem'),
            # Somme mensuelle des passages (basée sur les totaux quotidiens de passages)
            activity_count_sum_monthly=('activity_count_sum_daily', 'sum')
        ).reset_index()

        # Renommer les colonnes pour correspondre à ce que le graphique attend pour la durée moyenne et la somme des passages
        activity_monthly = activity_monthly.rename(columns={
            'duration_min_mean_of_daily_totals': 'duration_min_mean', # C'est maintenant la moyenne des totaux quotidiens
            'duration_min_sem_of_daily_totals': 'duration_min_sem',
            'activity_count_sum_monthly': 'activity_count_sum'      # Somme totale des passages pour le mois
        })

        # Calcul de activity_count_mean_daily (nombre moyen de passages par jour DANS LE MOIS)
        if pd.api.types.is_datetime64_any_dtype(activity_monthly['date']):
            activity_monthly['days_in_month'] = activity_monthly['date'].dt.daysinmonth
            activity_monthly['activity_count_mean_daily'] = np.where(
                activity_monthly['days_in_month'] > 0,
                activity_monthly['activity_count_sum'] / activity_monthly['days_in_month'], # passages totaux du mois / jours du mois
                0
            )
        else:
            activity_monthly['days_in_month'] = 0
            activity_monthly['activity_count_mean_daily'] = 0

        # Ajout du format MM/YY pour l'affichage des mois
        if 'date' in activity_monthly.columns and not activity_monthly.empty:
            activity_monthly['month_label'] = activity_monthly['date'].dt.strftime('%m/%y')
        else:
             activity_monthly['month_label'] = pd.Series(dtype='str')
    else:
        activity_monthly = pd.DataFrame(columns=['date', 'activity_count_sum', 'duration_min_mean',
                                                 'duration_min_sem', 'days_in_month',
                                                 'activity_count_mean_daily', 'month_label'])

    return activity_daily_for_graph, activity_monthly




# --- Figure Creation Function ---
def create_toilet_figure(daily_data, monthly_data, scale, selected_month):
    
    """Creates the Plotly figure for toilet activity based on inputs."""
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
        margin = MARGIN_CHART,
        hoverlabel=dict( # Configuration de l'infobulle (tooltip)
            font_size=16,                   
            font_color="white",          
            namelength=-1                   
        )
    )

    if scale == 'year':
        df = monthly_data
        if not df.empty:

            fig.add_trace(go.Bar(x=df['month_label'], y=df['duration_min_mean'], name="Durée moyenne (min)", error_y=dict(type='data', array=df['duration_min_sem']), marker_color=DATA1_COLOR))
            fig.add_trace(go.Scatter(x=df['month_label'], y=df['activity_count_sum'], name="Passages totaux", yaxis='y2', mode='lines+markers', line=dict(color=DATA2_COLOR)))
            
            if 'activity_count_mean_daily' in df.columns:
                fig.add_trace(go.Scatter(x=df['month_label'], y=df['activity_count_mean_daily'], name="Passages moyens/jour", yaxis='y2', mode='lines+markers', line=dict(color=DATA3_COLOR)))
            
            fig.update_layout(
                title=dict(text="Vue Annuelle : Activité mensuelle", font=dict(color=TEXT_COLOR),x = TITLE_X, y = TITLE_Y),
                xaxis=dict(title=dict(text="Mois", font=dict(color=TEXT_COLOR))),
                yaxis=dict(title=dict(text="Durée moyenne (min)", font=dict(color=DATA1_COLOR)), tickfont=dict(color=DATA1_COLOR)),
                yaxis2=dict(title=dict(text="Nombre de Passages", font=dict(color=DATA3_COLOR)), overlaying='y', side='right', tickfont=dict(color=DATA3_COLOR)),
                legend=LEGEND
            )
        else:
            fig.update_layout(title=dict(text="Yearly View: No monthly data available", font=dict(color=TEXT_COLOR)))

    elif scale == 'month' and selected_month:
        df = daily_data[daily_data['year_month'] == selected_month]
        if not df.empty:

             # Trace 1: Durée TOTALE par jour en minutes (Axe Y principal - gauche)
            if 'duration_min_sum' in df.columns: 
                fig.add_trace(go.Bar(x=df['date'], y=df['duration_min_sum'], name="Durée totale (min)", yaxis='y1', marker_color = DATAMONTH_COLOR))
            
            # Trace 2: Nombre de passages par jour (Axe Y secondaire - droite)
            fig.add_trace(go.Scatter(x=df['date'], y=df['activity_count_sum'],name="Passages par jour",yaxis='y2', mode='lines+markers', line=dict(color=DATAMONTH2_COLOR) ))

            fig.update_layout(
                title=dict(text=f"Vue Journalière : {selected_month}", font=dict(color=TEXT_COLOR), x=TITLE_X, y=TITLE_Y),
               xaxis=dict(title=dict(text="Jour", font=dict(color=TEXT_COLOR)), tickformat='%d',tickmode='array', tickvals= df['date'], ticktext= df['date'].dt.strftime('%d'), dtick="D1"),
                yaxis=dict(title=dict(text="Durée totale (min)", font=dict(color=DATAMONTH_COLOR)), tickfont=dict(color=DATAMONTH_COLOR),),
                yaxis2=dict(title=dict(text="Nombre de passages", font=dict(color=DATAMONTH2_COLOR)), tickfont=dict(color=DATAMONTH2_COLOR),overlaying='y', side='right',showgrid=False),
                legend=LEGEND
            )
        else:
            fig.update_layout(title=dict(text=f"Monthly View: No data available for {selected_month}", x=TITLE_X, y=TITLE_Y))
    elif scale == 'month' and not selected_month:
        fig.update_layout(title=dict(text="Monthly View: Please select a month", font=dict(color=TEXT_COLOR)))

    if not fig.data:
        fig.update_layout(title=dict(text="Aucune donnée à afficher", font=dict(color=TEXT_COLOR)))

    return fig



# --- Standalone App Execution ---
if __name__ == '__main__':
    # Load data when running standalone
    activity_daily_data, activity_monthly_data = get_toilet_data()
    available_months = sorted(activity_daily_data['year_month'].unique()) if not activity_daily_data.empty else []

    # Initialize the app
    app = Dash(__name__)
    app.title = APP_TITLE

    # Define the layout
    app.layout = html.Div(style={'backgroundColor': BACKGROUND_COLOR, 'color': TEXT_COLOR, 'padding': '20px'}, children=[
        html.H2(APP_TITLE, style={'textAlign': 'center', 'marginBottom': '30px'}),
        html.Div([
            html.Label("Select view scale:", style={'marginRight': '10px'}),
            dcc.RadioItems(
                id='scale-selector',
                options=[{'label': 'Year View (Monthly)', 'value': 'year'}, {'label': 'Month View (Daily)', 'value': 'month'}],
                value='year',
                labelStyle={'display': 'inline-block', 'marginRight': '20px'},
                inputStyle={'marginRight': '5px'}
            ),
        ], style={'marginBottom': '20px', 'textAlign': 'center'}),
        html.Div(id='month-dropdown-container', children=[
            html.Label("Select Month:", style={'marginRight': '10px'}),
            dcc.Dropdown(
                id='month-dropdown',
                options=[{'label': m, 'value': m} for m in available_months],
                value=available_months[0] if len(available_months) > 0 else None,
                clearable=False,
                style={'width': '200px', 'display': 'inline-block', 'color': '#333'}
            )
        ], style={'display': 'none', 'textAlign': 'center', 'marginBottom': '20px'}),
        dcc.Graph(id='activity-graph', style={'height': '60vh'})
    ])



    # Callback to show/hide month dropdown
    @app.callback(
        Output('month-dropdown-container', 'style'),
        Input('scale-selector', 'value')
    )
    def toggle_month_dropdown(scale):
        display = 'block' if scale == 'month' else 'none'
        return {'display': display, 'textAlign': 'center', 'marginBottom': '20px'}

    # Callback to update graph
    @app.callback(
        Output('activity-graph', 'figure'),
        Input('scale-selector', 'value'),
        Input('month-dropdown', 'value')
    )
    def update_graph_standalone(scale, selected_month):
        # Call the figure creation function using pre-loaded data
        return create_toilet_figure(activity_daily_data, activity_monthly_data, scale, selected_month)


    # Run the app
    app.run(debug=True)
