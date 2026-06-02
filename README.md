# IMC Prosperity 4 Go Tritons!!!
## Round 1 
### Algorithmic trading 

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

Then, we can optimize the price q by buying q=9999 at P=30 or q=4999 at P=28, because it will result in the order of bid price P=28, volume=40k. To optimize q, we can just buy q=9999 at P=30.
Same strategy can be applied to Ember Mushroom even though the order book is slightly more complicated. In this way, it turns out to be a math problem with a correct answer, and we also get full pnl for this.

## Round 2
### Algorithmic trading 

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
### Algorithmic trading 

### Manual Trading

## Round 4
### Algorithmic trading 

### Manual Trading

## Round 5
### Algorithmic trading 

### Manual Trading
