import random


def generate_artist_string():
    # 原始字符串列表
    string_list = [
        "anmi",
        "ask(askzy)",
        "hoji(hooooooooji1029)",
        "hiten_(hitenkei)",
        "mika_pikazo",
        "sheya",
        "tiv",
        "ciloranko",
        "akizero1510",
        "binggong_asylum",
        "cui_(jidanhaidaitang)",
        "ke-ta",
        "kuzuvine",
        "mamimi(mamamimi)",
        "mignon",
        "milkychu",
        "missile228",
        "miv4t",
        "nixeu",
        "asteroid_ill",
        "onineko",
        "kukka",
        "ningen_mame",
        "momoko(momopoco)",
        "you_shimizu",
        "cierra(ra-bit)",
        "mandrill",
        "atdan",
        "yoshinari_you",
        "liduke",
        "reoen",
        "huanxiang_heitu",
        "mochizuki_kei",
        "touzai(poppin phl95)",
        "namie",
        "kedama_milk",
        "kawacy",
        "cogecha",
        "wlop",
        "amazuyu_tatsuki",
        "yoneyama_mai",
        "konya_karasue",
        "rei(sanbonzakura)",
        "ogipote",
        "fuzichoco",
        "rella",
        "666pigeon",
        "akakura",
        "sho_(sho_lwlw)",
        "tianliang_duohe_fangdongye",
        "omone_hokoma_agm",
        "infukun",
        "chen_bin",
        "blade_(galaxist)",
    ]

    # 随机选择2到6个元素
    selected_elements = random.sample(string_list, 3)

    # 为每个元素生成一个随机数（0.3到0.9之间）
    result = ",".join(
        f"({element}:{random.uniform(0.3, 0.9)})" for element in selected_elements
    )
    # 如果wlop后面的数字大于0.4，则修改为0.4
    if "wlop" in result:
        if float(result.split("wlop:")[1].split(")")[0]) > 0.4:
            result = result.replace(
                "wlop:" + result.split("wlop:")[1].split(")")[0], "wlop:0.4"
            )
    return result


# Example usage
if __name__ == "__main__":
    print(generate_artist_string())
# [artist:alpha],artist:ciloranko,solo,[artist:sho (sho lwlw)],[[tianliang duohe fangdongye]],[artist:rhasta],year 2023,wlop,kani biimu,artist:kani_biimu,omochi_monaka,(artist:kedama milk:0.8),artist:ciloranko,artist:ke-ta,(artist:Hiten:0.8),artist:ciloranko,artist:ke-ta,(artist:Hiten:0.8),(artis1:nekoda:0.8),(artist:ask\(askzy\):0.8),(artist:As109:0.9),(yoneyama mai:1),(torino aqua:0.6),(piromizu:0.3),newest,(ogipote:1.1025), (bm tol:1.1025), (yoneyama mai:0.907), (ciloranko:0.907), (blue-senpai:0.907), aikome \(haikome\), artist:yoneyama mai,artist:ciloranko,artist:sho {sho lwlw},(toosaka asagi:1.1025), (ningen mame:0.907), (ciloranko:0.9524), sho \(sho lwlw\), (tianliang duohe fangdongye:0.8638), (rhasta:0.907), (sheya:0.2),(wlop:0.4),(sy4:1.1),(ciloranko:0.8),(ask \(askzy\):0.5),(remsrar:0.6),(cogecha:1.2),(artist:alpha:0.7),# [artist:alpha],artist:ciloranko,solo,[artist:sho (sho lwlw)],[[tianliang duohe fangdongye]],[artist:rhasta],year 2023,wlop,kani biimu,artist:kani_biimu,omochi_monaka,(artist:kedama milk:0.8),artist:ciloranko,artist:ke-ta,(artist:Hiten:0.8),artist:ciloranko,artist:ke-ta,(artist:Hiten:0.8),(artis1:nekoda:0.8),(artist:ask\(askzy\):0.8),(artist:As109:0.9),(yoneyama mai:1),(torino aqua:0.6),(piromizu:0.3),newest,(ogipote:1.1025), (bm tol:1.1025), (yoneyama mai:0.907), (ciloranko:0.907), (blue-senpai:0.907), aikome \(haikome\), artist:yoneyama mai,artist:ciloranko,artist:sho {sho lwlw},(toosaka asagi:1.1025), (ningen mame:0.907), (ciloranko:0.9524), sho \(sho lwlw\), (tianliang duohe fangdongye:0.8638), (rhasta:0.907), (sheya:0.2),(wlop:0.4),(sy4:1.1),(ciloranko:0.8),(ask \(askzy\):0.5),(remsrar:0.6),(cogecha:1.2),(artist:alpha:0.7),
