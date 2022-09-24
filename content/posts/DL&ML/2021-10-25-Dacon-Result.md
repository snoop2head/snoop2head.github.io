---
layout: single
title: "Assessment on the Dacon’s Stock Price Prediction Competition"


---

**🥈 NMAE score of 4.44 ranked the 2nd place / 29 teams**

- [🔗 Team's Github Repository](https://github.com/snoop2head/elastic-stock-prediction)
- [🔗 Dacon Stock Price Competition Page](https://dacon.io/competitions/official/235800/leaderboard)

![image-20211025181930856](../assets/images/2021-10-25-Dacon-Result/image-20211025181930856.png)

### Leaderboard Shake-up

There was huge shake-ups on the leaderboard. 
- public 1st → private 6th
- *public 2nd → private 2nd (🤚 Our team)*
- public 3rd  → private 12th
- public 4th → private 1st
- public 5th → private 3rd

Shake-up was due to China Evergrande debt crisis affecting Korean stock market. 
- The public leaderboard week of `09-15 ~ 09-24` was prior to the Evergrande debt crisis. 
- The private leaderboard week of `09-27 ~ 10-01` was just after Evergrande issue was disclosed.

Our team maintained the second place for both public and private leaderboard. **Therefore, it is reasonable to conclude that our ElasticNetCV model isn’t overfitted to particular week.**

### Performance

**Our ElasticNetCV model’s NMAE score outperformed Baseline’s score by  30%.** 

|       Model       | NMAE (09-06 ~ 09-10) | NMAE (09-27 ~ 10-01) |
| :---------------: | :----------------------------: | :----------------------------: |
|  **ElasticNetCV** |              **3.02**              |              **4.44**              |
| Linear Regression (Baseline) |              4.03              |               6.42               |

The submission is the result of ElasticNetCV model only. This was to assess the single model’s performance compared to other possibly ensembled submissions. Thus, we can possibly conclude that ElaticNetCV may reduce the errors of the prediction when ensembled with other model.

