# IMC Prosperity 4 Go Tritons!!!
## Round 1 
### Algorithmic Trading 

### Manual Trading
According to the auction rules, we can writing the following formulas for Dryland flax, where q is the quantity we are buying. We do not consider selling because the auction rules trivially make selling a loss. 

$$
P = 27: \quad V = \min\left\lbrace 75k + q,\ 0 \right\rbrace = 0
$$

$$
P = 28: \quad V = \min\left\lbrace 47k + q,\ 40k \right\rbrace = 40k
$$

$$
P = 29: \quad
V = \min\left\lbrace 35k + q,\ 40k \right\rbrace =
\begin{cases}
35k + q, & \text{if } q < 4999, \\
40k, & \text{otherwise.}
\end{cases}
$$

$$
P = 30: \quad
V = \min\left\lbrace 30k + q,\ 40k \right\rbrace =
\begin{cases}
30k + q, & \text{if } q < 9999, \\
40k, & \text{otherwise.}
\end{cases}
$$

Then, we can optimize the price q by buying q=9999 at P=30 or q=4999 at P=28, because it will result in the order of bid price P=28, volume=40k. To optimize q, we can just buy q=9999 at P=30. Same strategy can be applied to Ember Mushroom even though the order book is slightly more complicated (we bought q=19999 ember mushrooms at price=20). In this way, it turns out to be a math problem with a correct answer, and we also get full pnl for this.

## Round 2
### Algorithmic Trading 

### Manual Trading
According to the rules, we can set up a few functions to optimize the PnL. Let X(x) denote the function for research, Y(y) for scale, Z(z) for speed. Under the rules, we can write:

$$
\{PnL}(X, Y, Z) = XYZ - 500(x + y + z)
$$

$$
X(x) = 200{,}000 \cdot \frac{\log(x)}{\log(1 + 100)}
$$

$$
Y(y) = \frac{1}{100}y
$$

$$
Z(z) = z_{\alpha}
$$

$$
x+y+z≤100
$$

By calculus, we know that X and Y can be optimized, so the only issue is to find the value of z that gives a good balance. We wrote a simple simulation to guess how would other teams pick 
their z values, and it turned out z=37 would be able to give us a balanced between the hit rate and the pnl (higher z can increase the hit rate but also harms the total pnl significantly).
Subsequently, it gives x=16, y=47. We used up to 100% budget because the pnl formula implies every 1% used will increase the pnl by at least 500. Eventually, it turns out a pnl of 204,355, which is pretty decent.

## Round 3
### Algorithmic Trading 

### Manual Trading
Let's call the lowest bid b1 and the highest bid b2. Given that all products will be sold at price=920 next day, our team's strategy was to maximize the number of trades. Meanwhile, simple calculation shows that the penalty of low b2 is harsh. Therefore, we adjusted the simulation from round 2 and slightly increased the simluated value, leading us to set b2=876 (one extra dollar guarantees the trade to succeed). Then, we can use the information that the distribution of the bids is uniformly distributed at increments of 5 between 670 and 920. This helps use to determine b1. The expected bid price is 795. Since our b2 is relatively high, we adjusted b1 to 791. Eventually, it turned out that the average b2 among all teams is 859, so our guess was not too bad. 

## Round 4
### Algorithmic Trading 


### Manual Trading
For the manual trading challenge, we treated the problem as a static derivatives portfolio optimization problem under the given GBM model. We estimated each option’s fair value using the provided assumptions: zero drift, 251% annualized volatility, and discrete monitoring at 4 steps per trading day. Based on expected value, our core exotic trades were to buy `AC_45_KO`, sell `AC_40_BP`, and sell `AC_50_CO`.

To control risk, we added targeted hedges instead of fully hedging every exposure. We bought `AC_45_P` to hedge the short binary put, since it protects the region where the crystal finishes below 40. We also bought `AC_50_C`, `AC_50_C_2`, and `AC_50_P_2` to hedge the upside/downside risks from shorting the chooser option. We avoided extra options such as `AC_35_P`, `AC_40_P`, and `AC_60_C` because they either overlapped with existing hedges or increased overhedging risk.

Final portfolio: buy 500 `AC_45_KO`, sell 50 `AC_40_BP`, sell 50 `AC_50_CO`, buy 45 `AC_45_P`, buy 30 `AC_50_C`, buy 40 `AC_50_C_2`, and buy 50 `AC_50_P_2`. This portolio ended up with roughly 44.6k, (manual) round ranking 450. 

## Round 5
### Algorithmic Trading 
Our final algorithm used a hybrid trading framework that combined directional alpha, relative-value signals, trend following, and risk controls. Instead of applying one universal model to every product, we separated products into different strategy types based on their historical behavior. Products with clear directional drift were traded with high-conviction static or semi-static positions, while products with more stable relationships were handled using relative-value or pair-trading logic. For noisier products, we either reduced exposure or avoided trading them entirely unless there was a clear standalone signal. However, the result turned out to be a bit overfitting, because the signals are less responsive, leading to slower orders when the regime changed. Also, due to time constraint, we were not able to completely fix this issue, and the final result was lower than our expectation. 


### Manual Trading
In this round, we just read the news and make best possible decisions we can make. The following table is our portfolio:

| Tradable Good        | Direction | Allocation |
| -------------------- | --------: | ---------: |
| Obsidian cutlery     |      Sell |         0% |
| Lava cake            |      Sell |        13% |
| Magma ink            |       Buy |        15% |
| Volcanic incense     |       Buy |         6% |
| Sulfur reactor       |       Buy |        10% |
| Thermalite core      |       Buy |        15% |
| Scoria paste         |       Buy |        10% |
| Pyroflex cells       |      Sell |        10% |
| Ashes of the Phoenix |      Sell |         6% |

We only used up to 85% of the budget because the fee does not grow linearly. It turned out a total fee of 99.1k and a pnl around 51.5k (slightly worse than our expectation), most likely because of some misinterpretation in the news letter. 
