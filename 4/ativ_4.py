# pyright: reportUnusedExpression=false

# |%%--%%| <7ra0KERMts|lyFGxnQI9p>

# Importando bibliotecas
from tqdm import tqdm
import numpy as np
import pandas as pd
import socceraction.spadl as spd
from socceraction import xthreat as xt

# |%%--%%| <lyFGxnQI9p|K2BwC6JeSx>

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics

# |%%--%%| <K2BwC6JeSx|1CKaF8zuVd>
r"""°°°
# [CDAF] Atividade 4
°°°"""
# |%%--%%| <1CKaF8zuVd|bDCLDQ3BPV>
r"""°°°
## Nome e matrícula
Nome: Igor Lacerda Faria da Silva
Matrícula: 2020041973
°°°"""
# |%%--%%| <bDCLDQ3BPV|gsvwfbbfWQ>
r"""°°°
### LaLiga  p/ SPADL com pré-processamentos
°°°"""
# |%%--%%| <gsvwfbbfWQ|ISdgdbD1L1>

DATA_FOLDER = "data"

# |%%--%%| <ISdgdbD1L1|gSQIUK1GmU>

# Para o depurador...
# DATA_FOLDER = "../data/"

# |%%--%%| <gSQIUK1GmU|eYH2fDiFt6>

COUNTRY = "Spain"

# |%%--%%| <eYH2fDiFt6|ZRDSKmfUhL>

# carregando os eventos
# path = r"C:\Users\Galo\Hugo_Personal\Data\Wyscout_Top_5\events\events_Spain.json"
path = f"{DATA_FOLDER}/events/events_{COUNTRY}.json"
events = pd.read_json(path_or_buf=path)

# |%%--%%| <ZRDSKmfUhL|VyumaUqtiF>

# pré processamento em colunas da tabela de eventos para facilitar a conversão p/ SPADL
events = events.rename(
    columns={
        "id": "event_id",
        "eventId": "type_id",
        "subEventId": "subtype_id",
        "teamId": "team_id",
        "playerId": "player_id",
        "matchId": "game_id",
    }
)
events["milliseconds"] = events["eventSec"] * 1000
events["period_id"] = events["matchPeriod"].replace({"1H": 1, "2H": 2})

# |%%--%%| <VyumaUqtiF|YKqum5JZJj>

# carregando as partidas, pois vamos saber quais times jogam em casa e fora p/ usar como parametro do SPADL
# path = r"C:\Users\Galo\Hugo_Personal\Data\Wyscout_Top_5\matches\matches_Spain.json"
path = f"{DATA_FOLDER}/matches/matches_{COUNTRY}.json"
matches = pd.read_json(path_or_buf=path)

# |%%--%%| <YKqum5JZJj|VMQ4nP1dcr>

# as informações dos times de cada partida estão em um dicionário dentro da coluna 'teamsData', então vamos separar essas informações
team_matches = []
for i in tqdm(range(len(matches))):
    teams_data = matches.loc[i, "teamsData"]
    if isinstance(teams_data, dict):  # check if teams_data is a dictionary-like object
        match = pd.DataFrame(teams_data).T
        match["matchId"] = matches.loc[i, "wyId"]
        team_matches.append(match)
    else:
        # handle the case where teams_data is not a dictionary-like object
        print(f"teamsData for match {matches.loc[i, 'match_id']} is not a dictionary.")
team_matches = pd.concat(team_matches).reset_index(drop=True)

# |%%--%%| <VMQ4nP1dcr|CZfeg1CShd>

# fazendo a conversão p/ SPADL, padronizando a direção de jogo da esquerda p/ a direita e adicionando os nomes dos tipos de ações
actions = []
game_ids = events.game_id.unique().tolist()
for g in tqdm(game_ids):
    match_events = events.loc[events.game_id == g]
    match_home_id = team_matches.query(f"matchId == {g} and side == 'home'")[
        "teamId"
    ].values[0]
    match_actions = spd.wyscout.convert_to_actions(
        events=match_events, home_team_id=match_home_id
    )
    match_actions = spd.play_left_to_right(
        actions=match_actions, home_team_id=match_home_id
    )
    match_actions = spd.add_names(match_actions)
    actions.append(match_actions)
spadl = pd.concat(actions).reset_index(drop=True)

# |%%--%%| <CZfeg1CShd|9ljXuWlgrw>

# adicionando o nome dos jogadores
# path = r"C:\Users\Galo\Hugo_Personal\Data\Wyscout_Top_5\players.json"
path = f"{DATA_FOLDER}/players/players.json"
players = pd.read_json(path_or_buf=path)
players["player_name"] = players["shortName"].apply(
    lambda x: x.encode("utf-8").decode("unicode-escape")  # conserte as strings
)
players = players[["wyId", "player_name"]].rename(columns={"wyId": "player_id"})
spadl = spadl.merge(players, on="player_id", how="left")

# |%%--%%| <9ljXuWlgrw|VKVu0WiGyt>
r"""°°°
## Questão 1
- Crie um dataframe "shots" à partir do dataframe "spadl", contendo apenas os chutes.
- Crie 4 colunas no dataframe "shots" a serem usadas como features de um modelo de xG.
- Justifique a escolha das features.
°°°"""
# |%%--%%| <VKVu0WiGyt|YK5SD1DFF1>

spadl.info()

# |%%--%%| <YK5SD1DFF1|lzd8090FV4>

types_of_shot = ["shot", "shot_freekick", "shot_penalty"]
df_shots: pd.DataFrame = spadl.query("type_name in @types_of_shot")
df_shots

# |%%--%%| <lzd8090FV4|mJNtt7hPxt>

GOAL_CENTER_X: int = 105
GOAL_CENTER_Y: int = 34

UPPER_CROSSBAR_X: int = 105
UPPER_CROSSBAR_Y: int = 38

LOWER_CROSSBAR_X: int = 105
LOWER_CROSSBAR_Y: int = 30

# |%%--%%| <mJNtt7hPxt|cyjqzX1H9d>

df_shots["shot_distance"] = np.sqrt(
    (df_shots["start_x"] - GOAL_CENTER_X) ** 2
    + (df_shots["start_y"] - GOAL_CENTER_Y) ** 2
)

# |%%--%%| <cyjqzX1H9d|e72in88qQT>


def get_shot_angle(shot_pos_x, shot_pos_y):
    u = np.array([UPPER_CROSSBAR_X - shot_pos_x, UPPER_CROSSBAR_Y - shot_pos_y])
    v = np.array([LOWER_CROSSBAR_X - shot_pos_x, LOWER_CROSSBAR_Y - shot_pos_y])
    return np.arccos(np.dot(u / np.linalg.norm(u), v / np.linalg.norm(v)))


df_shots["shot_angle"] = df_shots[["start_x", "start_y"]].apply(
    lambda pos: get_shot_angle(pos["start_x"], pos["start_y"]), axis=1
)

# |%%--%%| <e72in88qQT|xYyvBDX9Wg>

df_shots["distance_x_angle"] = df_shots["shot_angle"] * df_shots["shot_distance"]

# |%%--%%| <xYyvBDX9Wg|gYP2FbZaKR>

df_shots["bodypart_weight"] = df_shots["bodypart_name"].apply(
    lambda x: 1 if x == "foot" else 0.3
)

# |%%--%%| <gYP2FbZaKR|Rwe9ZGjRBv>

df_shots.info()

# |%%--%%| <Rwe9ZGjRBv|wPTNZywTMu>
r"""°°°
### Escolhas

- Distância: um candidato óbvio, pois é muito mais fácil acertar chutes de perto.

- Ângulo: outro parâmetro clássico, sem muito o que falar. 

- `distance_x_angle`: aumentando o peso dos parâmetros tradicionais, peguei mais porque o Meira comentou em uma das aulas.

- `bodypart_weight`: pra fechar o time com chave de ouro, é reduzido o peso de partes do corpo que não são os pés, porque, presumivelmente, é mais difífil fazer gol com outras partes.
°°°"""
# |%%--%%| <wPTNZywTMu|fDg8yzZCFK>
r"""°°°
## Questão 2
- Crie uma coluna numérica binária "goal" no dataframe "shots" indicando se o chute resultou em gol ou não.
- Use regressão logística p/ treinar (.fit(X_train, y_train)) um modelo de xG usando as features criadas na questão 1.
- Use 70% dos dados para treino e 30% para teste.
- Reporte a acurácia do modelo para os conjuntos de treino (.score(X_train, y_train)) e teste (.score(X_test, y_test)).
°°°"""
# |%%--%%| <fDg8yzZCFK|VN08nih8oc>

# Não é necessário criar uma coluna nova, basta usar a coluna de result_id

# |%%--%%| <VN08nih8oc|7YiqfwIbYc>

x_train, x_test, y_train, y_test = train_test_split(
    df_shots[["shot_distance", "shot_angle", "distance_x_angle", "bodypart_weight"]],
    df_shots["result_id"],
    test_size=0.3,
)

# |%%--%%| <7YiqfwIbYc|ISRC6kljdP>

model = LogisticRegression()
model.fit(x_train, y_train)

# |%%--%%| <ISRC6kljdP|DXkZK9UYt2>

y_train_acc = model.score(x_train, y_train)
y_test_acc = model.score(x_test, y_test)

print(f"Acurácia nos dados de treino: {y_train_acc}")
print(f"Acurácia nos dados de teste: {y_test_acc}")

# |%%--%%| <DXkZK9UYt2|n4IS2mRKCA>

y_pred = model.predict(x_test)
cm = metrics.confusion_matrix(y_pred, y_test)
cm_display = metrics.ConfusionMatrixDisplay(
    confusion_matrix=cm, display_labels=["no goal", "goal"]
)
cm_display.plot()

# |%%--%%| <n4IS2mRKCA|oK1NilaAH8>
r"""°°°
## Questão 3
- Use o modelo treinado na questão 2 p/ prever a probabilidade de gol de todos os chutes do dataframe "shots". Reporte essas probabilidades no dataframe "shots" em uma coluna "xG".
- Agrupe o dataframe "shots" por "player_name" e reporte a soma dos "goal" e "xG".
- Reporte os 10 jogadores com maior xG.
- Reporte os 10 jogadores com maior diferença de Gols e xG.
°°°"""
# |%%--%%| <oK1NilaAH8|aKDODp8nxt>

probabilities = model.predict_proba(
    df_shots[["shot_distance", "shot_angle", "distance_x_angle", "bodypart_weight"]]
)
df_shots["xG"] = probabilities[:, 1]
df_shots["xG"]

# |%%--%%| <aKDODp8nxt|WdexzZLRR5>

columns = ["result_id", "xG"]
shots_by_player = df_shots.groupby(["player_name"])[columns].sum()
shots_by_player

# |%%--%%| <WdexzZLRR5|E0WBlB0DTJ>

shots_by_player.sort_values("xG", ascending=False).head(10)

# |%%--%%| <E0WBlB0DTJ|bF04UleYMa>

shots_by_player["diff"] = shots_by_player["result_id"] - shots_by_player["xG"]

# |%%--%%| <bF04UleYMa|3gBTtlbRpu>

shots_by_player.sort_values("diff", ascending=False)[columns].head(10)

# |%%--%%| <3gBTtlbRpu|XQ150xYhxw>
r"""°°°
## Questão 4
- Instancie um objeto ExpectedThreat com parâmetros l=25 e w=16.
- Faça o fit do modelo ExpectedThreat com o dataframe "spadl".
°°°"""
# |%%--%%| <XQ150xYhxw|ykmZuVHRI8>

xT = xt.ExpectedThreat(l=25, w=16)
# workaround for socceraction v1.4.0
xt._safe_divide = lambda a, b: np.nan_to_num(a / b)
xT.fit(spadl)

# |%%--%%| <ykmZuVHRI8|alvrNx7zSg>
r"""°°°
## Questão 5
- Crie um dataframe "prog_actions" à partir do dataframe "spadl", contendo apenas as ações de progressão e que são bem-sucedidas.
- Use o método rate do objeto ExpectedThreat p/ calcular o valor de cada ação de progressão do dataframe "prog_actions", em uma coluna chamada "action_value".
- Agrupe o dataframe "prog_actions" por "player_name" e reporte a soma dos "action_value".
- Reporte os 10 jogadores com maior "action_value".
°°°"""
# |%%--%%| <alvrNx7zSg|nsWkC0TDFx>

prog_actions = xt.get_successful_move_actions(spadl)
prog_actions["action_value"] = xT.rate(prog_actions)

# |%%--%%| <nsWkC0TDFx|WypxX7D61v>

action_value = prog_actions.groupby("player_name")["action_value"].sum()

# |%%--%%| <WypxX7D61v|8E4c6JZ24c>

action_value.nlargest(n=10)
