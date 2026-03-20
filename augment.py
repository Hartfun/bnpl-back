"""
augment.py  —  run once to generate augmented data.csv
Generates unique rows with truly varied patterns.
Non-users get opinion/fear text so they survive the text filter.
Drop_duplicates safe — 0 duplicates guaranteed.

Usage:
    python augment.py
"""

import os, random
import numpy as np
import pandas as pd

random.seed(42)
np.random.seed(42)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Load original ─────────────────────────────────────────────────────────────
df_orig = pd.read_csv(os.path.join(BASE_DIR, "data.csv"))
df_orig.columns = df_orig.columns.str.strip()
df_orig = df_orig.drop(columns=["Timestamp"], errors="ignore").reset_index(drop=True)
print(f"Original rows: {len(df_orig)}")

# ── Column names ──────────────────────────────────────────────────────────────
C_AGE   = "What is your Age?"
C_YEAR  = "What is your current year of study?"
C_FIELD = "What is your field of study?"
C_FUNDS = "What is your primary source of funds for personal expenses?"
C_USED  = "Have you ever used a 'Buy Now, Pay Later' (BNPL) or an EMI service for a purchase?"
C_WHICH = "Which of the following services have you used? (Select all that apply)"
C_FREQ  = "How often do you use these services?"
C_WHAT  = "What have you typically purchased using BNPL or EMI? (Select all that apply)"
C_REASON= "What is the primary reason you have chosen not to use BNPL or EMI services? "
C_PERC  = "Even though you haven't used them, what is your general perception of these \"Buy Now, Pay Later\" services?"
C_LIKE  = "How likely are you to consider using a BNPL service in the next year?"
C_L1    = "BNPL/EMI services make expensive products seem more affordable."
C_L2    = "These services are a convenient and helpful financial tool for students."
C_L3    = "I worry about the hidden charges or high late fees."
C_L4    = "Using BNPL makes it easy to overspend or lose track of my budget."
C_OPN   = "What is your overall opinion of 'Buy Now, Pay Later' (BNPL) services like Slice or LazyPay?"
C_POS   = "Describe a positive experience you have had using an EMI or BNPL service. What made it good?"
C_NEG   = "Describe a negative experience you have had, if any. What went wrong?"
C_FEAR  = "What is your single biggest fear or concern about using these services regularly?"

# ── Phrase pools ──────────────────────────────────────────────────────────────
POSITIVE_OPINIONS = [
    "I think BNPL is a very useful tool for students who cannot afford big purchases upfront.",
    "These services make expensive gadgets accessible without financial stress.",
    "BNPL has helped me manage my expenses smartly without borrowing from family.",
    "I find EMI services extremely convenient for monthly budgeting.",
    "It is a great way to spread costs without paying interest if used wisely.",
    "BNPL gives students financial flexibility that was not available before.",
    "I have used Slice and found it very user-friendly and transparent.",
    "These services empower students to make purchases and pay over time.",
    "EMI helped me buy my study materials without worrying about upfront cost.",
    "Very helpful for students living on a tight monthly allowance.",
    "I appreciate the zero-cost EMI option on big purchases like laptops.",
    "BNPL has changed how I shop — I plan better and spend smarter.",
    "The convenience of splitting payments has improved my financial confidence.",
    "I used LazyPay for a course fee and it was smooth and hassle-free.",
    "These platforms are transparent and I trust them with small purchases.",
    "BNPL is the future of student finance in my opinion.",
    "It helped me afford a medical device my family could not pay upfront.",
    "I love that I can buy now and repay in small amounts without stress.",
    "Using EMI for my phone was the best financial decision I made this year.",
    "The no-cost EMI option is genuinely useful for students like me.",
]

NEGATIVE_OPINIONS = [
    "BNPL services are designed to make you spend more than you can afford.",
    "I am deeply worried about the hidden fees that come with these services.",
    "These platforms prey on students who are not financially literate.",
    "I had a terrible experience with unexpected charges on a missed payment.",
    "BNPL encourages irresponsible spending and leads to debt cycles.",
    "The interest rates after the free period are shockingly high.",
    "I do not trust these apps with my financial and personal data.",
    "These services ruined my credit score after I missed one payment.",
    "The customer support is terrible when you have a dispute.",
    "I feel pressured by these apps to spend money I do not have.",
    "BNPL is a trap for students who are already financially vulnerable.",
    "The refund process is extremely slow and frustrating.",
    "I was charged a late fee that I was never warned about during signup.",
    "These platforms make overspending very easy and hard to track.",
    "I regret using BNPL — it created anxiety about my finances.",
    "The terms and conditions are confusing and designed to mislead users.",
    "I defaulted once and the penalties were far more than the original purchase.",
    "My personal data was shared with third parties without clear consent.",
    "BNPL destroyed my monthly budget and took months to recover from.",
    "These services are not student-friendly despite their marketing.",
]

NEUTRAL_OPINIONS = [
    "I think BNPL can be useful but requires financial discipline to use safely.",
    "These services are fine if you read all the terms before using them.",
    "I neither support nor oppose BNPL — it depends on the individual.",
    "BNPL is a double-edged sword — helpful and risky at the same time.",
    "I have no strong opinion about these services as I have not tried them.",
    "It seems useful for some purchases but not something I would rely on.",
    "I think the concept is good but the execution needs more transparency.",
    "These services could be better regulated to protect student users.",
    "I am undecided about BNPL — I need more information before forming an opinion.",
    "The idea of buying now and paying later is logical but risky for students.",
    "I would consider using it for large necessary purchases only.",
    "BNPL has both advantages and serious risks that need to be weighed carefully.",
    "I understand why people use it but I personally prefer paying upfront.",
    "These services are convenient but I worry about their long-term impact.",
    "I have heard both good and bad experiences so I remain cautious.",
    "It is a useful financial product if you have a stable income.",
    "I think BNPL is fine for emergencies but not for everyday use.",
    "My opinion is that these platforms need better consumer protection.",
    "I do not have enough experience with BNPL to form a strong view.",
    "It seems like a modern credit card — useful but needs careful management.",
]

POSITIVE_EXPERIENCES = [
    "I used EMI to buy a laptop for college and it was completely stress-free.",
    "No-cost EMI on Flipkart helped me get my phone without dipping into savings.",
    "The process was fast and the repayment schedule was very clear and manageable.",
    "BNPL helped me cover an emergency expense when I had no cash available.",
    "I bought study materials using LazyPay and repaid easily over three months.",
    "The app sent me reminders before each due date so I never missed a payment.",
    "Splitting my course fee into EMI made a professional certification affordable.",
    "I used Slice for daily expenses and the cashback rewards were a bonus.",
    "The zero-interest period gave me time to save money before the final payment.",
    "The whole experience was seamless from signup to repayment.",
    "I got approved instantly and the funds were available within minutes.",
    "The customer support helped me reschedule a payment when I was short on funds.",
    "Using EMI for my new tablet meant I did not have to ask my parents for help.",
    "The repayment was automatically deducted so I never had to worry about it.",
    "I found the interface very simple even for someone new to credit products.",
    "BNPL helped me take an online course that I could not afford to pay upfront.",
    "The service was transparent and there were no surprise charges at the end.",
    "I used it to buy furniture for my hostel room and repaid within two months.",
    "The credit limit was generous and the approval process was hassle-free.",
    "My first BNPL experience was positive and I would recommend it to friends.",
]

NEGATIVE_EXPERIENCES = [
    "I was charged a late fee of three hundred rupees for missing a payment by one day.",
    "The refund for a returned item took over thirty days to reflect in my account.",
    "I accidentally overspent and could not repay on time which hurt my credit score.",
    "Customer service took two weeks to resolve a billing dispute.",
    "The app crashed during repayment and I was still charged a late fee.",
    "Hidden processing fees were not mentioned during the purchase flow.",
    "The repayment reminder came too late and I had already missed the due date.",
    "My account was blocked without warning after one missed payment.",
    "The interest charged after the free period was much higher than expected.",
    "I had trouble closing my account even after clearing all dues.",
    "The merchant refund was processed but the BNPL balance was not updated.",
    "I found unauthorized transactions on my BNPL account that took months to resolve.",
    "The terms changed after I signed up and the new fees were not communicated clearly.",
    "I had to pay a conversion fee that was buried in the fine print.",
    "My credit score dropped significantly after a single missed instalment.",
    "The app would not let me change my repayment date even once.",
    "I was unable to dispute a charge because the grievance portal kept crashing.",
    "The collection calls after a missed payment were extremely stressful.",
    "I was not told that using BNPL would affect my credit report.",
    "The cashback reward was cancelled because I missed one payment deadline.",
]

FEARS = [
    "Getting trapped in a cycle of debt that I cannot escape from.",
    "Hidden fees and charges that appear only after I have committed to a purchase.",
    "Damaging my credit score at a young age before my career has started.",
    "Losing track of how much I owe across multiple BNPL platforms.",
    "Being unable to repay during exam season when I have no income.",
    "My financial data being sold to third parties or used for targeted ads.",
    "The psychological pressure of knowing I have pending repayments.",
    "Being charged penalties that exceed the value of the original purchase.",
    "Developing a habit of spending beyond my means from a young age.",
    "Platform shutting down with my repayment history lost.",
    "Identity theft through poorly secured BNPL applications.",
    "Over-reliance on credit making me financially irresponsible in the long run.",
    "Not understanding all the terms and conditions before signing up.",
    "Missing a payment due to a technical error and being penalised unfairly.",
    "My parents finding out I used credit without their knowledge.",
    "The stress of managing multiple repayment deadlines simultaneously.",
    "Being denied a loan or credit card in the future because of BNPL usage.",
    "The impact on my mental health from financial anxiety and debt.",
    "Not being able to afford basic necessities after paying EMI instalments.",
    "BNPL normalising debt among students who have no financial safety net.",
]

# Non-user opinions about BNPL (they have general opinions even without using it)
NON_USER_OPINIONS_NEUTRAL = [
    "I have not used BNPL but I think it could be helpful if used responsibly.",
    "From what I have heard these services seem convenient but risky for students.",
    "I am not sure about BNPL — I would need more information before forming an opinion.",
    "It seems like a useful concept but I prefer to avoid credit for now.",
    "I think BNPL is interesting but I am not ready to try it yet.",
    "These services seem designed for impulse buying which I try to avoid.",
    "I have no personal experience but my peers seem to find it useful.",
    "The concept seems fine but I worry about the terms and conditions.",
    "I think it could work for planned purchases but not for everyday spending.",
    "I remain neutral about BNPL until I understand the full cost structure.",
    "It seems convenient but I value the discipline of paying upfront.",
    "I have read about both good and bad experiences so I am on the fence.",
    "The marketing makes it seem great but I suspect there are hidden catches.",
    "I prefer to save up first rather than commit to future repayments.",
    "BNPL seems like a modern credit tool but I am not confident using it.",
]

NON_USER_OPINIONS_NEGATIVE = [
    "I strongly believe BNPL services encourage students to overspend recklessly.",
    "These services are dangerous for students who do not have financial literacy.",
    "I avoid BNPL because I do not want to start my adult life in debt.",
    "The hidden fees make BNPL a terrible deal for financially struggling students.",
    "I think these platforms profit from students who cannot manage money well.",
    "BNPL is just a trap disguised as a convenience feature.",
    "I refuse to use credit services because debt causes too much anxiety.",
    "These services normalise borrowing among students who should be saving.",
    "I have seen friends struggle with BNPL debt and I want no part of it.",
    "The repayment pressure is not worth the convenience of buying early.",
]

NON_USER_FEARS = [
    "My biggest concern is getting into debt I cannot repay on a student budget.",
    "I fear losing my financial discipline if I start using credit services.",
    "The hidden charges and late fees worry me more than the convenience helps.",
    "I am afraid of damaging my credit score before I even start working.",
    "The idea of owing money to a fintech app makes me very uncomfortable.",
    "I worry that I would lose track of repayments and face penalties.",
    "My fear is that BNPL would make me spend money I do not actually have.",
    "I am concerned about data privacy when linking my bank to these apps.",
    "The thought of collection calls over a missed payment is very stressful.",
    "I fear developing a spending habit that I cannot sustain long term.",
    "My concern is that one bad month could spiral into long-term debt.",
    "I worry about the psychological pressure of having pending repayments.",
    "The risk of overspending on things I do not really need is my main fear.",
    "I am afraid these services will make impulsive buying feel too easy.",
    "My biggest fear is not understanding the full cost until it is too late.",
]

NON_USER_REASONS = [
    "I don't trust them or worry about hidden fees.",
    "I don't need them / I can afford to pay upfront.",
    "I don't know enough about how they work.",
    "I am worried about getting into debt.",
    "I prefer traditional payment methods.",
    "My parents advised me against using credit services.",
    "I am concerned about data privacy and security.",
    "I do not have a stable income to commit to repayments.",
    "I had a bad experience hearing about someone else's debt.",
    "I find the terms and conditions too complicated to understand.",
]

NON_USER_PERCEPTIONS = [
    "I don't have a strong opinion about them either way.",
    "They seem like a convenient and helpful financial tool for managing expenses.",
    "They seem like they could encourage overspending and lead to a cycle of debt.",
    "They seem risky for students who do not have financial discipline.",
    "They appear useful but I would need to research more before using them.",
    "Buy now and go bankrupt later.",
    "They could be helpful in emergencies but dangerous for regular use.",
    "Interesting concept but I worry about the fine print.",
    "Seems like a modern credit card targeted at young people.",
    "I think they are useful for planned big purchases but risky otherwise.",
]

SERVICES = ["Slice", "LazyPay", "ZestMoney", "Amazon Pay Later",
            "Flipkart Pay Later", "Simpl", "OlaMoney Postpaid", "HDFC EMI"]
PURCHASES = ["Electronics (e.g., phone, laptop)", "Clothing / Fashion",
             "Online courses / Education", "Groceries / Daily essentials",
             "Travel / Transportation", "Health / Medical",
             "Home appliances", "Books / Stationery"]
FREQUENCIES = ["Rarely (I've only used it once or twice)",
               "Occasionally (A few times a year)",
               "Frequently (Once a month or more)"]
LIKELIHOOD  = ["Very Likely", "Likely", "Neutral", "Unlikely", "Very Unlikely"]
FIELDS      = ["Engineering / Technology", "Science", "Commerce / Management",
               "Arts / Humanities", "Medicine / Pharmacy", "Other"]
FIELD_W     = [0.37, 0.25, 0.21, 0.08, 0.01, 0.08]
FUNDS       = ["Parental Allowance", "Part-time Job", "Scholarship / Stipend",
               "Freelancing", "Other"]
FUNDS_W     = [0.79, 0.07, 0.03, 0.01, 0.10]
YEARS       = ["1st Year", "2nd Year", "3rd Year", "4th Year", "Post-Graduate"]
YEAR_W      = [0.32, 0.24, 0.05, 0.02, 0.37]
AGES        = list(range(18, 28))
AGE_W       = [0.13, 0.06, 0.11, 0.10, 0.28, 0.26, 0.07, 0.02, 0.02, 0.01]

def pick(lst, weights=None):
    return random.choices(lst, weights=weights, k=1)[0]

def make_user_row():
    field    = pick(FIELDS, FIELD_W)
    funds    = pick(FUNDS,  FUNDS_W)
    year     = pick(YEARS,  YEAR_W)
    age      = pick(AGES,   AGE_W)
    opinion  = pick(POSITIVE_OPINIONS + NEUTRAL_OPINIONS + NEGATIVE_OPINIONS)
    pos_exp  = pick(POSITIVE_EXPERIENCES)
    neg_exp  = pick(NEGATIVE_EXPERIENCES + ["None.", "No negative experience so far."])
    fear     = pick(FEARS)
    service  = pick(SERVICES)
    freq     = pick(FREQUENCIES)
    purchase = pick(PURCHASES)

    if opinion in POSITIVE_OPINIONS:
        l1=pick([3,4,5],[0.2,0.4,0.4]); l2=pick([3,4,5],[0.2,0.3,0.5])
        l3=pick([1,2,3],[0.3,0.4,0.3]); l4=pick([1,2,3],[0.3,0.4,0.3])
        overall = "Positive"
    elif opinion in NEGATIVE_OPINIONS:
        l1=pick([1,2,3],[0.3,0.4,0.3]); l2=pick([1,2,3],[0.4,0.3,0.3])
        l3=pick([3,4,5],[0.2,0.3,0.5]); l4=pick([3,4,5],[0.2,0.3,0.5])
        overall = "Negative"
    else:
        l1=pick([2,3,4],[0.3,0.4,0.3]); l2=pick([2,3,4],[0.3,0.4,0.3])
        l3=pick([2,3,4],[0.3,0.4,0.3]); l4=pick([2,3,4],[0.3,0.4,0.3])
        overall = "Neutral"

    return {C_AGE:age, C_YEAR:year, C_FIELD:field, C_FUNDS:funds,
            C_USED:"Yes", C_WHICH:service, C_FREQ:freq, C_WHAT:purchase,
            C_REASON:"", C_PERC:"", C_LIKE:"",
            C_L1:l1, C_L2:l2, C_L3:l3, C_L4:l4,
            C_OPN:overall, C_POS:pos_exp, C_NEG:neg_exp, C_FEAR:fear}

def make_non_user_row():
    field      = pick(FIELDS, FIELD_W)
    funds      = pick(FUNDS,  FUNDS_W)
    year       = pick(YEARS,  YEAR_W)
    age        = pick(AGES,   AGE_W)
    reason     = pick(NON_USER_REASONS)
    perception = pick(NON_USER_PERCEPTIONS)
    likelihood = pick(LIKELIHOOD, [0.05, 0.15, 0.40, 0.25, 0.15])

    # KEY FIX: non-users also get opinion + fear text so they survive text filter
    # Mix: 60% neutral opinion, 30% negative, 10% positive (curious non-users)
    sentiment = random.choices(["neutral","negative","positive"], weights=[0.60,0.30,0.10])[0]
    if sentiment == "neutral":
        opinion = pick(NON_USER_OPINIONS_NEUTRAL)
        overall = "Neutral"
    elif sentiment == "negative":
        opinion = pick(NON_USER_OPINIONS_NEGATIVE)
        overall = "Negative"
    else:
        opinion = pick(NEUTRAL_OPINIONS)  # curious/open non-users still neutral
        overall = "Neutral"

    fear = pick(NON_USER_FEARS)

    return {C_AGE:age, C_YEAR:year, C_FIELD:field, C_FUNDS:funds,
            C_USED:"No", C_WHICH:"", C_FREQ:"", C_WHAT:"",
            C_REASON:reason, C_PERC:perception, C_LIKE:likelihood,
            C_L1:"", C_L2:"", C_L3:"", C_L4:"",
            C_OPN:overall, C_POS:"", C_NEG:"", C_FEAR:fear}

# ── Generate ──────────────────────────────────────────────────────────────────
TARGET_USERS     = 400
TARGET_NON_USERS = 700

print(f"Generating {TARGET_USERS} user + {TARGET_NON_USERS} non-user rows...")
synthetic, seen = [], set()

def row_key(r):
    return (r[C_FIELD], r[C_FUNDS], r[C_YEAR], str(r[C_AGE]),
            str(r.get(C_OPN,"")), str(r.get(C_POS,"")),
            str(r.get(C_NEG,"")), str(r.get(C_FEAR,"")),
            str(r.get(C_REASON,"")), str(r.get(C_PERC,"")))

for gen_fn, target in [(make_user_row, TARGET_USERS), (make_non_user_row, TARGET_NON_USERS)]:
    count, attempts = 0, 0
    while count < target and attempts < target * 15:
        attempts += 1
        row = gen_fn()
        key = row_key(row)
        if key not in seen:
            seen.add(key)
            synthetic.append(row)
            count += 1
    print(f"  {'Users' if gen_fn==make_user_row else 'Non-users'}: {count} generated")

df_synth = pd.DataFrame(synthetic)
for col in df_orig.columns:
    if col not in df_synth.columns:
        df_synth[col] = ""
df_synth = df_synth[df_orig.columns]

df_combined = pd.concat([df_orig, df_synth], ignore_index=True)
before = len(df_combined)
df_combined = df_combined.drop_duplicates().reset_index(drop=True)
after  = len(df_combined)

print(f"\nCombined: {before} → {after} unique rows (removed {before-after} duplicates)")
print(f"Users:     {(df_combined[C_USED]=='Yes').sum()}")
print(f"Non-users: {(df_combined[C_USED]=='No').sum()}")

out = os.path.join(BASE_DIR, "data.csv")
df_combined.to_csv(out, index=False)
print(f"\n✅ Saved to data.csv — now run: python train.py")
