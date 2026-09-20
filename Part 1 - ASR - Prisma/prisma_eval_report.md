# Gnani Prisma ASR Evaluation Report
_Generated: 2026-09-19 22:45:20 IST_

---

## Summary — Results by Category
| Category | n | Accuracy % | WER % | CER % | Subs% | Dels% | Ins% | Avg Latency | P50 Latency | P95 Latency | Min/Max Latency | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| accent_bengali | 20 | 84.0% | 16.9% | 8.1% | 12.8% | 0.0% | 4.1% | 0.55s | 0.50s | 0.78s | 0.34s / 0.91s |  |
| accent_malayalam | 20 | 84.6% | 20.4% | 7.5% | 14.5% | 0.6% | 5.3% | 0.68s | 0.68s | 0.91s | 0.34s / 0.96s |  |
| baseline_clean | 25 | 74.5% | 25.5% | 19.8% | 5.5% | 19.9% | 0.0% | 1.06s | 1.06s | 1.21s | 0.83s / 1.31s |  |
| code_switch * | 2 | 47.4% | 52.6% | 51.4% | 51.7% | 0.9% | 0.0% | 1.93s | 1.19s | 2.68s | 1.19s / 2.68s | 23 skipped |
| domain_medical | 25 | 86.1% | 13.9% | 4.2% | 8.2% | 1.7% | 4.0% | 0.89s | 0.89s | 1.33s | 0.57s / 1.43s |  |
| numeric | 25 | 95.0% | 5.0% | 2.4% | 3.8% | 1.2% | 0.0% | 0.88s | 0.85s | 1.27s | 0.63s / 1.32s |  |

_\* fewer than 10 scored clips — average less reliable._
_Latency = exact wall-clock time of each clip's successful API call (retry backoff excluded); averaged per category above._

---

## Subcategory Breakdowns

### numeric
| Subcategory | n | Accuracy % | WER % | CER % | Avg Latency |
|---|---|---|---|---|---|
| cardinal * | 5 | 98.8% | 1.2% | 0.5% | 0.83s |
| currency * | 5 | 93.8% | 6.2% | 2.0% | 0.89s |
| date * | 5 | 96.1% | 3.9% | 0.8% | 0.86s |
| ordinal | 10 | 93.1% | 6.9% | 4.3% | 0.92s |

---

## Worst 3 Clips per Category

### accent_bengali
- **accent_bengali_0001** — WER 118.2%
  - **Ref:** book a hp gas cylinder using my registered contact number 79083174897
  - **Hyp:** bouquet hp gas cylinder using my registered contact number 7 9 0 8 3 1 7 4 8 9 7
- **accent_bengali_0011** — WER 100.0%
  - **Ref:** moby dick
  - **Hyp:** movie t
- **accent_bengali_0013** — WER 100.0%
  - **Ref:** house
  - **Hyp:** haan

### accent_malayalam
- **accent_malayalam_0018** — WER 200.0%
  - **Ref:** 39
  - **Hyp:** 30 9
- **accent_malayalam_0016** — WER 100.0%
  - **Ref:** visual
  - **Hyp:** vishva
- **accent_malayalam_0004** — WER 26.7%
  - **Ref:** embroidery and there we use to teach us the basic embroidery patterns like running stage
  - **Hyp:** and there we they used to teach us the basic embroidery patterns like running stitch

### baseline_clean
- **AMI_EN2002b_H00_FEO070_0042617_0044950** — WER 100.0%
  - **Ref:** he he d he doesnt have to know exactly what i mean yeah m i mean its hot finished but it doesnt really say
  - **Hyp:** --
- **AMI_TS3003b_H01_MTD011UID_0059215_0060963** — WER 77.8%
  - **Ref:** uh uh ive do ive uh done a little uh research on the internet and not much information about it um about uh interface but uh uh
  - **Hyp:** doing a little research on the internet
- **AMI_EN2002d_H01_FEO072_0044186_0045922** — WER 76.2%
  - **Ref:** yeah i started i started using your um um the speak i dont know if youve seen that in the yeah
  - **Hyp:** try that a few times i started using your i dont know

### code_switch
- **audio_971.wav** — WER 100.0%
  - **Ref:** is worth the read i think put it like that im just thinking aloud another way of thinking of that is theres something called sturgeons law which basically it was first i think meant for science fiction but it applies to everything eventually and sturgeons law is that ninetyfive percent of everything is crap
  - **Hyp:** इस वरथ द रड लइक दट आई ऍम जसट थकग अलउड अनदअर व ऑफ थकग ऑफ दट इस दर इस समथग कलड सटरजनस ल वहच बसकल इट वस फरसट आई थक मनट फर सइस फकशन बट इट अपलइज ट एवरथग इवचअल एड सटरजन ल इस दट नइनटन फइव परसट ऑफ एवरथग इस सकरप
- **audio_1449.wav** — WER 5.3%
  - **Ref:** त लक जल ह इतहस खद क दहरन वल ह एक वचन क लए व लक जलन वल ह
  - **Hyp:** त लक चल ह इतहस खद क दहरन वल ह एक वचन क लए व लक जलन वल ह

### domain_medical
- **/240907/365FE17E-E6B5-4809-9739-FC7F5EBE1B4F/2.m4a** — WER 66.7%
  - **Ref:** zerodol and calpol
  - **Hyp:** 0 doll and calpol
- **/250427/ce-73d4588f-143b-4ffb-8944-a8df7f811c16/7.m4a** — WER 31.6%
  - **Ref:** not having adequate rest okay okay so that continues for better part of the day hmm hmm hmm then
  - **Hyp:** not having adequate rest so that continues for better part of the day
- **/240907/D3A675F9-69E5-4389-8092-984929C87834/2.m4a** — WER 30.8%
  - **Ref:** 500 mg also because youre feeling weak take zincovit once in a day
  - **Hyp:** 5 100 mg also because you are feeling weak take zincovit once in a day

### numeric
- **121-127105-0024** — WER 22.9%
  - **Ref:** poor douglas before his death when it was in sight committed to me the manuscript that reached him on the third of these days and that on the same spot with immense effect he began to read to our hushed little circle on the night of the fourth
  - **Hyp:** poor douglas before his death when it was in sight committed to me the manuscript that reached him on the third of these days and that on the same spot with immense effect he began to read
- **7729-102255-0028** — WER 14.3%
  - **Ref:** private persons who had leased the free state hotel vainly besought the various authorities to prevent the destruction of their property
  - **Hyp:** private persons who at least the free state hotel vainly besought the various authorities to present the destruction of their property
- **8463-294825-0015** — WER 12.5%
  - **Ref:** gram roughly 1 20 eighth of an ounce
  - **Hyp:** graham roughly 1 20 eighth of an ounce
