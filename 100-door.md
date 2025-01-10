# 100 Door Puzzle

## 问题重述
> There are 100 doors in a long hallway. They are all closed. The first time you walk by each door, you open it. The second time around, you close every second door (since they are all opened). On the third pass, you stop at every third door and open it if it’s closed, close it if it’s open. On the fourth pass, you take action on every fourth door. You repeat this pattern for 100 passes.
> **Question:** At the end of 100 passes, what doors are opened and what doors are closed?

## 分析与解答
这个问题没想通的时候觉得不难但是很繁，想通了觉得既不难也不繁。主要的想法是：某编号的门如果被经过了奇数次，则门的状态与起始状态相反；如果被经过了偶数次，则门的状态与起始状态相同。

那么通用的做法是看该门编号有多少个小于趟数的因数。如果有奇数个因数，则判定门的状态为起始状态的相反状态；否则，判定门的状态为起始状态。

具体到这道题，要不要这么做呢？wait... 马克思爷爷曾经教导我们说：“具体问题具体分析”。这个问题有什么具体情况呢？那就是：本题的趟数是和门数相同的。也就是说，只要看门的编号的因数个数就行了，不必关心“因数要小于等于趟数”这个限制条件。继续分析，某个数的成对因数对应于两趟，而这两趟下来对门的状态没有影响。那么，对门的状态有影响的就只有单因数情况，而只有完全平方数有单因数。这样，问题的解就是那些编号为完全平方数的门啦！！！

自此，答案显而易见，状态有变的门的编号为：1，4，9，16，25，36，49，64，81，100。

## 参考文献
1. [Door Toggling Puzzle Or 100 Doors Puzzle](http://classic-puzzles.blogspot.com/2008/05/door-toggling-puzzle-or-100-doors.html)
2. [PUZZLE – 100 DOORS](http://www.theodorenguyen-cao.com/2008/02/02/puzzle-100-doors/)

*写于 2015 年 9 月*