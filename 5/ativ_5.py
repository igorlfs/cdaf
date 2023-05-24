import numpy as np
import pandas as pd
import sklearn.metrics as mt
import socceraction.spadl as spd
import socceraction.vaep.formula as fm
import socceraction.vaep.labels as lab
import xgboost as xgb
from socceraction.vaep import features as ft
from tqdm import tqdm

# |%%--%%| <BJPjA64UOW|HcdimilNSO>
r"""°°°
# [CDAF] Atividade 5
°°°"""
# |%%--%%| <HcdimilNSO|hLAizgR43A>
r"""°°°
## Nome e matrícula
Nome: Igor Lacerda Faria da Silva
Matrícula: 2020041973
°°°"""
# |%%--%%| <hLAizgR43A|eTEN6LlswT>
r"""°°°
## Referências
- [1] https://tomdecroos.github.io/reports/kdd19_tomd.pdf
- [2] https://socceraction.readthedocs.io/en/latest/api/vaep.html
- [3] https://socceraction.readthedocs.io/en/latest/documentation/valuing_actions/vaep.html
- [4] https://github.com/ML-KULeuven/socceraction/tree/master/public-notebooks
°°°"""
# |%%--%%| <eTEN6LlswT|UfZk8nA4aW>
r"""°°°
## Introdução
- Nessa atividade, temos implementada a pipeline inteira do VAEP [1] para os dados do Wyscout das Top 5 ligas.
- [2] é a documentação das funções do VAEP na API do socceraction.
- [3] apresenta uma explicação do framework com uma mistura de intuição, matemática e código.
- [4] são notebooks públicos que implementam o VAEP para outro conjunto de dados.
°°°"""
# |%%--%%| <UfZk8nA4aW|3X0En2XvDE>
r"""°°°
## Instruções
- Para cada header do notebook abaixo, vocês devem explicar o que foi feito e à qual seção/subseção/equação do paper "Actions Speak Louder than Goals: Valuing Actions by Estimating Probabilities" ela corresponde. Justifique suas respostas.
- Além disso, após algumas partes do código haverão perguntas que vocês devem responder, possivelmente explorando minimamente o que já está pronto.
- Por fim, vocês devem montar um diagrama do fluxo de funções/tarefas de toda a pipeline do VAEP abaixo. Esse diagrama deve ser enviado como arquivo na submissão do Moodle, para além deste notebook.
°°°"""
# |%%--%%| <3X0En2XvDE|XQkZNMWco6>
r"""°°°
### Carregando os dados
°°°"""
# |%%--%%| <XQkZNMWco6|gaWTmOmxs2>


def load_matches(path):
    matches = pd.read_json(path_or_buf=path)
    # as informações dos times de cada partida estão em um dicionário dentro da coluna 'teamsData', então vamos separar essas informações
    team_matches = []
    for i in range(len(matches)):
        match = pd.DataFrame(matches.loc[i, "teamsData"]).T
        match["matchId"] = matches.loc[i, "wyId"]
        team_matches.append(match)
    team_matches = pd.concat(team_matches).reset_index(drop=True)

    return team_matches


# |%%--%%| <gaWTmOmxs2|2TDG5hFAVC>


def load_players(path):
    players = pd.read_json(path_or_buf=path)
    players["player_name"] = players["firstName"] + " " + players["lastName"]
    players = players[["wyId", "player_name"]].rename(columns={"wyId": "player_id"})

    return players


# |%%--%%| <2TDG5hFAVC|aoFY2dqlZL>


def load_events(path):
    events = pd.read_json(path_or_buf=path)
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

    return events


# |%%--%%| <aoFY2dqlZL|2qJhyNiPoy>


def load_minutes_played_per_game(path):
    minutes = pd.read_json(path_or_buf=path)
    minutes = minutes.rename(
        columns={
            "playerId": "player_id",
            "matchId": "game_id",
            "teamId": "team_id",
            "minutesPlayed": "minutes_played",
        }
    )
    minutes = minutes.drop(["shortName", "teamName", "red_card"], axis=1)

    return minutes


# |%%--%%| <2qJhyNiPoy|DEqAn2f7yV>

BASE_DIR = "data"

# |%%--%%| <DEqAn2f7yV|u9ATh7fwz4>

leagues = ["England", "Spain"]
events = {}
matches = {}
minutes = {}
for league in leagues:
    path = f"{BASE_DIR}/matches/matches_{league}.json"
    matches[league] = load_matches(path)
    path = f"{BASE_DIR}/events/events_{league}.json"
    events[league] = load_events(path)
    path = f"{BASE_DIR}/minutes_played/minutes_played_per_game_{league}.json"
    minutes[league] = load_minutes_played_per_game(path)

# |%%--%%| <u9ATh7fwz4|skKxnSZR0C>

path = f"{BASE_DIR}/players.json"
players = load_players(path)
players["player_name"] = players["player_name"].str.decode("unicode-escape")

# |%%--%%| <skKxnSZR0C|IipevwMpMf>
r"""°°°
#### Análise

Eu diria que esse trecho faz referência à subseção 2.1 do artigo, pois consiste em um pré-processamento para carregar os dados, que depois vão ser convertidos no formato SPADL. Esse trecho explora como os formatos de dados de diferentes provedoras *não são* uniformes, e mostra como é o pré-processamento de dados do Wyscout.
°°°"""
# |%%--%%| <IipevwMpMf|7GU6Nij1Sk>
r"""°°°
### SPADL
°°°"""
# |%%--%%| <7GU6Nij1Sk|TlXRMEToNl>


def spadl_transform(events, matches):
    spadl = []
    game_ids = events.game_id.unique().tolist()
    for g in tqdm(game_ids):
        match_events = events.loc[events.game_id == g]
        match_home_id = matches.loc[
            (matches.matchId == g) & (matches.side == "home"), "teamId"
        ].values[0]
        match_actions = spd.wyscout.convert_to_actions(
            events=match_events, home_team_id=match_home_id
        )
        match_actions = spd.play_left_to_right(
            actions=match_actions, home_team_id=match_home_id
        )
        match_actions = spd.add_names(match_actions)
        spadl.append(match_actions)
    spadl = pd.concat(spadl).reset_index(drop=True)

    return spadl


# |%%--%%| <TlXRMEToNl|odtvnbIgXx>

spadl = {}
for league in leagues:
    spadl[league] = spadl_transform(events=events[league], matches=matches[league])

# |%%--%%| <odtvnbIgXx|s7ZIbSv0AK>
r"""°°°
#### Análise

Esse trecho faz referência à seção 2.2, pois os dados são transformados no formato SPADL.
°°°"""
# |%%--%%| <s7ZIbSv0AK|GwYf5oViDW>
r"""°°°
### Features
°°°"""
# |%%--%%| <GwYf5oViDW|V8Upp1U1cs>


def features_transform(spadl):
    spadl.loc[spadl.result_id.isin([2, 3]), ["result_id"]] = 0
    spadl.loc[spadl.result_name.isin(["offside", "owngoal"]), ["result_name"]] = "fail"

    xfns = [
        ft.actiontype_onehot,
        ft.bodypart_onehot,
        ft.result_onehot,
        ft.goalscore,
        ft.startlocation,
        ft.endlocation,
        ft.team,
        ft.time,
        ft.time_delta,
    ]

    features = []
    for game in tqdm(np.unique(spadl.game_id).tolist()):
        match_actions = spadl.loc[spadl.game_id == game].reset_index(drop=True)
        match_states = ft.gamestates(actions=match_actions)
        match_feats = pd.concat([fn(match_states) for fn in xfns], axis=1)
        features.append(match_feats)
    features = pd.concat(features).reset_index(drop=True)

    return features


# |%%--%%| <V8Upp1U1cs|qnzGlbX73m>
r"""°°°
1- O que a primeira e a segunda linhas da função acima fazem? Qual sua hipótese sobre intuito dessas transformações? Como você acha que isso pode impactar o modelo final?
°°°"""
# |%%--%%| <qnzGlbX73m|o2C5LjnjHz>
r"""°°°
#### Resposta

No formato SPADL, existem mais de dois tipo de resultado para uma ação. Isto é, o resultado pode ser sucesso, falha ou outra coisa, que, no geral, pode ser considerada falha. Desse modo, o resultado dessas ações é convertido para falha. Creio que impedimentos são um tanto que neutros para um time (ao menos não positivos), e fazer gols contra nem se fala, então, a princípio, não vejo como essa transformação pode ser ruim. Imagino que isso não deve causar grandes impactos no modelo final, porque esses resultados são bastante incomuns.
°°°"""
# |%%--%%| <o2C5LjnjHz|isy658XzVU>

features = {}
for league in ["England", "Spain"]:
    features[league] = features_transform(spadl[league])

# |%%--%%| <isy658XzVU|8FPGvvkwl5>
r"""°°°
#### Análise

Esse trecho faz alusão à subseção 4.2 do artigo, em que são feitos ajustes nos dados para que o desempenho dos modelos seja melhor.
°°°"""
# |%%--%%| <8FPGvvkwl5|RxLHqKEgg8>
r"""°°°
### Labels
°°°"""
# |%%--%%| <RxLHqKEgg8|C1lfwTjL9K>


def labels_transform(spadl):
    yfns = [lab.scores, lab.concedes]

    labels = []
    for game in tqdm(np.unique(spadl.game_id).tolist()):
        match_actions = spadl.loc[spadl.game_id == game].reset_index(drop=True)
        labels.append(pd.concat([fn(actions=match_actions) for fn in yfns], axis=1))

    labels = pd.concat(labels).reset_index(drop=True)

    return labels


# |%%--%%| <C1lfwTjL9K|pHuJ6k81e2>

labels = {}
for league in ["England", "Spain"]:
    labels[league] = labels_transform(spadl[league])

# |%%--%%| <pHuJ6k81e2|tQPa5j9R65>

labels["England"]["scores"].sum()

# |%%--%%| <tQPa5j9R65|RIus8NBNKl>

labels["England"]["concedes"].sum()

# |%%--%%| <RIus8NBNKl|NJv73A6tmv>
r"""°°°
2- Explique o por que da quantidade de labels positivos do tipo scores ser muito maior que do concedes. Como você acha que isso pode impactar o modelo final?
°°°"""
# |%%--%%| <NJv73A6tmv|AXOmKnRaO1>
r"""°°°
#### Resposta

Geralmente, a maioria das ações que os jogadores fazem, tem como objetivo aumentar a chance de fazer gols. Dessa maneira, o esperado é que existam menos ações em que o efeito contrário é atingido. Isso impacta fortemente o treinamento do modelo, uma vez que ele pode ficar enviesado para labels positivos do tipo *scores*.
°°°"""
# |%%--%%| <AXOmKnRaO1|ecTJVvkmLO>
r"""°°°
#### Análise

Neste trecho, são construídas as labels da seção 4.1. Isso é bem sugestivo pelo `yfns` da função `labels_transform`.
°°°"""
# |%%--%%| <ecTJVvkmLO|QVAs4CfEjJ>
r"""°°°
### Training Model
°°°"""
# |%%--%%| <QVAs4CfEjJ|kSvTgyHOTq>


def train_vaep(X_train, y_train, X_test, y_test):
    models = {}
    for m in ["scores", "concedes"]:
        models[m] = xgb.XGBClassifier(random_state=0, n_estimators=50, max_depth=3)

        print("training " + m + " model")
        models[m].fit(X_train, y_train[m])

        p = sum(y_train[m]) / len(y_train[m])
        base = [p] * len(y_train[m])
        y_train_pred = models[m].predict_proba(X_train)[:, 1]
        train_brier = mt.brier_score_loss(
            y_train[m], y_train_pred
        ) / mt.brier_score_loss(y_train[m], base)
        print(m + " Train NBS: " + str(train_brier))
        print()

        p = sum(y_test[m]) / len(y_test[m])
        base = [p] * len(y_test[m])
        y_test_pred = models[m].predict_proba(X_test)[:, 1]
        test_brier = mt.brier_score_loss(y_test[m], y_test_pred) / mt.brier_score_loss(
            y_test[m], base
        )
        print(m + " Test NBS: " + str(test_brier))
        print()

        print("----------------------------------------")

    return models


# |%%--%%| <kSvTgyHOTq|nYpsaX76jJ>

models = train_vaep(
    X_train=features["England"],
    y_train=labels["England"],
    X_test=features["Spain"],
    y_test=labels["Spain"],
)

# |%%--%%| <nYpsaX76jJ|3RXs7t54XU>
r"""°°°
3- Por que treinamos dois modelos diferentes? Por que a performance dos dois é diferente?
°°°"""
# |%%--%%| <3RXs7t54XU|qtX0EhYBiB>
r"""°°°
#### Resposta

As ações que favorecem um gol não necessariamente desfavorecem um gol do time inimigo, isto é, essas probabilidades não são complementares. Estranhamente, mesmo com menos dados, o `concedes` tem uma performance melhor. Isso deve acontecer pois as ações que aumentam a chance de levar gol são mais bem definidas (possuem uma variância menor) do que as ações que levam o time a fazer gols.
°°°"""
# |%%--%%| <qtX0EhYBiB|lWq5IxppV9>
r"""°°°
#### Análise

Esse cabeçalho faz reverência à seção 4, em que é apresentada discussão do cálculo do VAEP e os algoritmos.
°°°"""
# |%%--%%| <lWq5IxppV9|KJrfrH1Zy3>
r"""°°°
### Predictions
°°°"""
# |%%--%%| <KJrfrH1Zy3|NwjetmT2y1>


def generate_predictions(features, models):
    preds = {}
    for m in ["scores", "concedes"]:
        preds[m] = models[m].predict_proba(features)[:, 1]
    preds = pd.DataFrame(preds)

    return preds


# |%%--%%| <NwjetmT2y1|HClRRXr88n>

preds = {}
preds["Spain"] = generate_predictions(features=features["Spain"], models=models)
preds["Spain"]

# |%%--%%| <HClRRXr88n|ghQVdZwJ8T>
r"""°°°
#### Análise

A seção 5, no geral, explora a performance do modelo, avaliando diferentes estatísticas, tal qual as previsões são geradas neste cabeçalho.
°°°"""
# |%%--%%| <ghQVdZwJ8T|lBH1OivT7Q>
r"""°°°
### Action Values
°°°"""
# |%%--%%| <lBH1OivT7Q|ukDIEn6C96>


def calculate_action_values(spadl, predictions):
    action_values = fm.value(
        actions=spadl, Pscores=predictions["scores"], Pconcedes=predictions["concedes"]
    )
    action_values = pd.concat(
        [
            spadl[
                [
                    "original_event_id",
                    "action_id",
                    "game_id",
                    "start_x",
                    "start_y",
                    "end_x",
                    "end_y",
                    "type_name",
                    "result_name",
                    "player_id",
                ]
            ],
            predictions.rename(columns={"scores": "Pscores", "concedes": "Pconcedes"}),
            action_values,
        ],
        axis=1,
    )

    return action_values


# |%%--%%| <ukDIEn6C96|XuO6zYgq9R>

action_values = {}
action_values["Spain"] = calculate_action_values(
    spadl=spadl["Spain"], predictions=preds["Spain"]
)
action_values["Spain"]

# |%%--%%| <XuO6zYgq9R|6JyvMHmD0v>

valuable_actions = action_values["Spain"].query("Pscores >= 0.95")
valuable_actions

# |%%--%%| <6JyvMHmD0v|jd6ERYc0f1>
r"""°°°
4- Explore as ações com Pscores >= 0.95. Por que elas tem um valor tão alto? As compare com ações do mesmo tipo e resultado opostado. Será que o modelo aprende que essa combinação de tipo de ação e resultado está diretamente relacionado à variável y que estamos tentando prever?

5- Qual formula do paper corresponde à coluna `offensive_value` do dataframe `action_values`? E a coluna `defensive_value`?
°°°"""
# |%%--%%| <jd6ERYc0f1|1d5iZK8TT4>
r"""°°°
#### Respostas

4) Todas as ações com $P_{scores} \geq 0.95$ são chutes (ao gol), que, no geral, incrementam muito a chance de se fazer gol. Por outro lado, quando essas ações dão errado, elas diminuem muito a chance de se marcar. Outras ações não sofrem desse viés: essa alteração drástica ocorre exclusivamente com chutes. Dessa maneira, essas outras ações acabam não sendo tão valorizadas.

5) Às equações 1 e 2, respectivamente.
°°°"""
# |%%--%%| <1d5iZK8TT4|lm1xoWHkpf>
r"""°°°
#### Análise

Não sei avaliar a qual seção este código em particular se refere, uma vez que ele estende a tabela para conter os dados do trecho anterior.
°°°"""
# |%%--%%| <lm1xoWHkpf|iW31CNT88F>
r"""°°°
### Player Ratings
°°°"""
# |%%--%%| <iW31CNT88F|eR4m86DzE3>


def calculate_minutes_per_season(minutes_per_game):
    minutes_per_season = minutes_per_game.groupby("player_id", as_index=False)[
        "minutes_played"
    ].sum()

    return minutes_per_season


# |%%--%%| <eR4m86DzE3|23x9VDdjaf>

minutes_per_season = {}
minutes_per_season["Spain"] = calculate_minutes_per_season(minutes["Spain"])

# |%%--%%| <23x9VDdjaf|XsA4cWzaZ4>


def calculate_player_ratings(action_values, minutes_per_season, players):
    player_ratings = (
        action_values.groupby(by="player_id", as_index=False)
        .agg({"vaep_value": "sum"})
        .rename(columns={"vaep_value": "vaep_total"})
    )
    player_ratings = player_ratings.merge(
        minutes_per_season, on=["player_id"], how="left"
    )
    player_ratings["vaep_p90"] = (
        player_ratings["vaep_total"] / player_ratings["minutes_played"] * 90
    )
    player_ratings = (
        player_ratings[player_ratings["minutes_played"] >= 600]
        .sort_values(by="vaep_p90", ascending=False)
        .reset_index(drop=True)
    )
    player_ratings = player_ratings.merge(players, on=["player_id"], how="left")
    player_ratings = player_ratings[
        ["player_id", "player_name", "minutes_played", "vaep_total", "vaep_p90"]
    ]

    return player_ratings


# |%%--%%| <XsA4cWzaZ4|s4sTsYRC2y>

player_ratings = {}
player_ratings["Spain"] = calculate_player_ratings(
    action_values=action_values["Spain"],
    minutes_per_season=minutes_per_season["Spain"],
    players=players,
)
player_ratings["Spain"].nlargest(5, "vaep_p90")

# |%%--%%| <s4sTsYRC2y|8sKuyQiUyI>
r"""°°°
6- Acha que o Top 5 da lista é bem representativo? Compare esse ranqueamento do VAEP com o do xT da Atividade 4. Qual você acha que é mais representativo?
°°°"""
# |%%--%%| <8sKuyQiUyI|IwTyj2y9Ej>
r"""°°°
#### Resposta

Com toda certeza esse top 5 é representativo, todos os jogadores são excepcionais. Eu diria que ele é mais representativo que o Top 10 da atividade 4. Como apresentado no artigo, o VAEP tende a ser uma métrica mais acurada para avaliar jogadores em diferentes contextos (tendo como base, por exemplo, o valor dos jogadores).
°°°"""
# |%%--%%| <IwTyj2y9Ej|f3mByInik7>
r"""°°°
#### Análise

Este trecho final corresponde à subseção 5.5, como é evidente pelas avaliações dos jogadores.
°°°"""
