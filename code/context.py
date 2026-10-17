from __future__ import annotations

CONTEXT_V = 1

BAND_MAX = {1: 5, 2: 6, 3: 6}

PART_TITLES = {
    1: "第一部分：回答問題 (Part I: Answering Questions)",
    2: "第二部分：表達意見 (Part II: Expressing Opinions)",
    3: "第三部分：整合式寫作 (Part III: Writing an Integrated Essay)",
}

RUBRIC_ITEMS = {
    1: "切題度、語法、詞彙、拼字、標點使用",
    2: "切題度、組織連貫性、語法、詞彙",
    3: "論述切題度、組織連貫性、語法、詞彙",
}

LANGUAGE_CONSTRUCT = {
    1: "詢問與回答個人相關的資訊；說明、描述個人經驗及與生活及學習相關的資訊",
    2: "表達自己的看法（喜歡/不喜歡、同意/不同意）；與人交換意見（提供建議）",
    3: "說明圖文內容重點並表達看法；摘要描述重點、比較關鍵訊息；表達立場、闡述意見，並提出適當的援引佐證",
}

RUBRIC_BANDS = {
    1: (
        "5級分：能充分符合題目要求。能掌握基本的句型結構和詞彙，錯誤不影響語意表達。拼字、標點正確性高。",
        "4級分：能符合題目要求。能大致掌握基本的句型結構，語法仍有錯誤。詞彙有限，大致不妨礙語意表達。拼字、標點錯誤偏多。",
        "3級分：能大致符合題目要求。能大致掌握基本的句型結構，語法仍有錯誤。詞彙有限，大致不妨礙語意表達。拼字、標點錯誤偏多。",
        "2級分：能大致符合題目要求。語法錯誤多，語句多呈片段。詞彙非常有限，阻礙語意表達。拼字、標點錯誤非常多。",
        "1級分：僅部分符合題目要求。語法錯誤多，語句多呈片段。詞彙非常有限，阻礙語意表達。拼字、標點錯誤非常多。",
        "0級分：作答量明顯不足（或未答），或表達內容不切題（等同未答）。語法錯誤過多，無法清楚表達。",
    ),
    2: (
        "6級分：能充分符合題目要求。組織完整，能靈活使用連接詞和轉折詞，表達流暢、清楚有條理。能適切使用語法、詞彙，鮮有錯誤。",
        "5級分：能符合題目要求。組織良好，能使用適當的連接詞和轉折詞，表達通順、有條理。語法有變化，使用上雖有錯誤，但無礙表達。詞彙足供表達。",
        "4級分：能符合題目要求。組織大致良好，能使用基本的連接詞，使文意連貫。對於較複雜的概念常需迂迴表達。能掌握基本的句型結構，錯誤不影響語意表達，標點正確性高。能掌握基本的詞彙，拼字正確性高。",
        "3級分：能大致符合題目要求。組織大致良好，能使用基本的連接詞，使文意連貫。對於較複雜的概念常需迂迴表達。能掌握基本的句型結構，錯誤不影響語意表達，標點正確性高。能掌握基本的詞彙，拼字正確性高。",
        "2級分：能大致符合題目要求。表達費力，需耐心解讀。僅能使用最基本的連接詞。能大致掌握基本的句型結構，但時有錯誤，影響語意表達，標點錯誤偏多。詞彙有限，拼字錯誤偏多。",
        "1級分：僅部分符合題目要求。表達費力，需耐心解讀。僅能使用最基本的連接詞。能大致掌握基本的句型結構，但時有錯誤，影響語意表達，標點錯誤偏多。詞彙有限，拼字錯誤偏多。",
        "0級分：作答量不足（或未答），或表達內容不切題（等同未答）。語法錯誤多，詞彙非常有限，無法清楚表達。",
    ),
    3: (
        "6級分：能充分符合題目要求。組織完整，能靈活使用連接詞和轉折詞，表達流暢、清楚有條理。能適切使用語法、詞彙，鮮有錯誤。",
        "5級分：能符合題目要求。組織良好，能使用適當的連接詞和轉折詞，表達通順、有條理。語法有變化，使用上雖有錯誤，但無礙表達。詞彙足供表達。",
        "4級分：能大致符合題目要求。組織良好，能使用適當的連接詞和轉折詞，表達通順、有條理。語法有變化，使用上雖有錯誤，但無礙表達。詞彙足供表達。",
        "3級分：能大致符合題目要求。組織大致良好，能使用基本的連接詞使文意連貫。對於較複雜的概念常需迂迴表達。能掌握基本的句型結構，錯誤不影響語意表達，標點正確性高。能掌握基本的詞彙，拼字正確性高。",
        "2級分：僅部分符合題目要求。組織大致良好，能使用基本的連接詞使文意連貫。對於較複雜的概念常需迂迴表達。能掌握基本的句型結構，錯誤不影響語意表達，標點正確性高。能掌握基本的詞彙，拼字正確性高。",
        "1級分：僅部分符合題目要求。表達費力，需耐心解讀。僅能使用最基本的連接詞。能大致掌握基本的句型結構，但時有錯誤，影響語意表達，標點錯誤偏多。詞彙有限，拼字錯誤偏多。",
        "0級分：作答量不足（或未答），或表達內容不切題（等同未答）。語法錯誤過多，詞彙非常有限，無法清楚表達。",
    ),
}

TASK_TEXT = {
    (1, 1): (
        "Read the information below and provide answers using approximately 25 words in total.\n"
        "--- Poster ---\n"
        "Campus Guides Needed. Give tours of Southern University to high school seniors.\n"
        "Training: Saturday, July 15. One tour per day on August 9, 16 & 23.\n"
        "Call 2283-9976 for information.\n"
        "--- Questions ---\n"
        "Question 1. How many tours will be given?\n"
        "Question 2. Ask the organizer one question to find out more about the training.\n"
        "Question 3. Describe the place in your school that you visit most often. Use full sentences."
    ),
    (2, 1): (
        "Read the information below and provide answers using approximately 25 words in total.\n"
        "--- Poster ---\n"
        "Welcome New Students! Music / Games / Barbecue @ Jolly Beach.\n"
        "Leave campus at 6 p.m. on July 31. Return at midnight.\n"
        "--- Questions ---\n"
        "Question 1. How long does the activity take?\n"
        "Question 2. Ask the organizer one question about the fee for this activity.\n"
        "Question 3. Describe a similar activity you have attended with some classmates. Use full sentences."
    ),
    (1, 2): (
        "Write a short text using approximately 80 words.\n"
        "You are Lynn Pan. Your university has recently opened a new food court on campus. "
        "It includes a vegetarian restaurant, two tea shops, and a well-known fast food restaurant. "
        "You decided to express your opinion on this new addition to campus.\n"
        "In your email (To: universityfood@nstu.edu, Subject: Some ideas, addressed to the Administrator), you should\n"
        "1. explain whether you think this is a good change;\n"
        "2. make at least one suggestion about how the food service can be improved on your campus."
    ),
    (2, 2): (
        "Write a short text using approximately 80 words.\n"
        "You are Dorothy Wang. Your university allows cars on campus. However, several accidents "
        "involving cars have recently occurred on campus. You have decided to write an email to "
        "your school's administrator.\n"
        "In your email (To: admin@ncu.edu, Subject: Some suggestions, addressed to the Administrator), you should\n"
        "1. express your concern about this issue;\n"
        "2. give at least one suggestion on how to solve the problem."
    ),
    (1, 3): (
        "Write a short essay using approximately 120-150 words.\n"
        "In class, your professor presents two figures during her talk on career opportunities "
        "after graduation for students in your field. She then gives you an assignment on the "
        "following topic:\n"
        "How do students better prepare themselves for joining the workforce after graduation?\n"
        "In the assignment, you should write a short essay in which you\n"
        "1. summarize the main points shown in the two figures;\n"
        "2. state your own opinion on the topic, using the information from the two figures "
        "as well as citing examples from your own experience."
    ),
    (2, 3): (
        "Write a short essay using approximately 120-150 words.\n"
        "In class, your professor presents two figures during her talk on helping students "
        "achieve a better study-life balance. She then gives you an assignment on the "
        "following topic:\n"
        "What adjustments can students make to achieve a better study-life balance?\n"
        "In the assignment, you should write a short essay in which you\n"
        "1. summarize the main points shown in the two figures;\n"
        "2. state your own opinion on the topic, using the information from the two figures "
        "as well as citing examples from your own experience."
    ),
}

GRADING_TEMPLATE = """You are an official BESTEP (基本英語能力測驗) writing examiner grading one essay.

== Scoring rubric — {part_title} ==
評分項目：{rubric_items}
評分量表（表現說明）：
{rubric_bands}

== Deliverable ==
You will output exactly ONE integer band score (級分) from 0 to {band_max} for the essay below. Output the single digit only.

== Writing task (Form {form}) ==
{task_text}

== Candidate's essay ==
{essay}

Band score: """


def build_grading_prompt(form: int, part: int, essay: str) -> str:
    if part not in BAND_MAX:
        raise ValueError(f"part must be 1, 2, or 3 — got {part!r}")
    if (form, part) not in TASK_TEXT:
        raise ValueError(f"unknown (form, part) = ({form!r}, {part!r})")
    return GRADING_TEMPLATE.format(
        part_title=PART_TITLES[part],
        rubric_items=RUBRIC_ITEMS[part],
        rubric_bands="\n".join(RUBRIC_BANDS[part]),
        band_max=BAND_MAX[part],
        form=form,
        task_text=TASK_TEXT[(form, part)],
        essay=str(essay).strip(),
    )
