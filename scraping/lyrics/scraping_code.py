import lyricsgenius 
import pandas as pd
import langdetect
from tqdm import tqdm

genius = lyricsgenius.Genius("FyCz1Ocv_IgTnIhd66Dk8F7UGxvGmsih6_li4W4CWFUbWWID51TxvXeWZ6deWEXk", verbose=2)


songs = pd.read_csv("data/songs_scraped_new_scraped.csv")
n = 0
songs["haslyrics"] =  ~songs["lyrics"].isna()
for index, song in songs.iterrows():
    
    if not song["haslyrics"]:# and not song["nosongfound"]:
        song_gen = genius.search_song(song["song_mod"], song["artist_mod"])
        print(song["song_mod"])
        print(song["artist_mod"])
        if(song_gen):    
            songs.at[index,"lyrics"] = song_gen.lyrics
        else:
            songs.at[index,"nosongfound"] = True
        n += 1
        songs.to_csv("data/songs_scraped_new_scraped.csv", index=False)
        if n > 1000:
            print(index)
            break



