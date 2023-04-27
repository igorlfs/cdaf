#!/bin/bash

# fill in data directory
mkdir -p \
	data/competitions \
	data/events \
	data/matches \
	data/playeranks \
	data/teams \
	data/players \
	data/referees \
	data/coaches \
	data/tagId2name \
	data/eventId2name

# downloads the datasets stored in https://figshare.com/collections/Soccer_match_event_dataset/4415000/5
# already unzip content and rename the files to be more readable
wget -P data/competitions https://figshare.com/ndownloader/files/15073685
mv data/competitions/15073685 data/competitions/competitions.json

wget -P data/events https://figshare.com/ndownloader/files/14464685
unzip data/events/14464685 -d data/events/
rm data/events/14464685

wget -P data/matches https://figshare.com/ndownloader/files/14464622
unzip data/matches/14464622 -d data/matches/
rm data/matches/14464622

wget -P data/playeranks https://figshare.com/ndownloader/files/16972010
mv data/playeranks/16972010 data/playeranks/playeranks.json

wget -P data/players https://figshare.com/ndownloader/files/15073721
mv data/players/15073721 data/players/players.json

wget -P data/teams https://figshare.com/ndownloader/files/15073697
mv data/teams/15073697 data/teams/teams.json

wget -P data/referees https://figshare.com/ndownloader/files/15074030
mv data/referees/15074030 data/referees/referees.json

wget -P data/coaches https://figshare.com/ndownloader/files/15073868
mv data/coaches/15073868 data/coaches/coaches.json

wget -P data/tagId2name https://figshare.com/ndownloader/files/21385239
mv data/tagId2name/21385239 data/tagId2name/tagId2name.csv

wget -P data/eventId2name https://figshare.com/ndownloader/files/21385245
mv data/eventId2name/21385245 data/eventId2name/eventId2name.csv
