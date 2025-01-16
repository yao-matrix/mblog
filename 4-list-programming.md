# 四道有趣的单链表编程题

链表也算是基本数据类型之一了。记得刚学习 C 语言的时候，链表和数组是见得最多的数据类型了。也正因为它的常用，在编程面的时候自然会有所涉及，本文主要讨论四道我觉得比较有趣的关于单链表的面试题。 
- Q1：链表的反序
- Q2：找出链表的中间元素
- Q3：链表排序
- Q4：判断一个单链表是否有环

以下给出链表结点的数据结构：

```C++
typedef struct _list_node {
    double keyVal;
    struct _list_node *next;
} ListNode;
```

## Q1：单链表的反序

```C++
ListNode* reverseList(ListNode* head) {
    ListNode *p1, *p2 , *p3;

    // 链表为空，或是单结点链表直接返回头结点
    if (head == NULL || head->next == NULL) {
        return head;
    }

    p1 = head;
    p2 = head->next;

    while (p2 != NULL) {
        p3 = p2->next;
        p2->next = p1;
        p1 = p2;
        p2 = p3;
    }

    head->next = NULL;
    head = p1;

    return head;
}
```

## Q2：找出链表的中间元素

```C++
ListNode* find_midlist(ListNode* head) {
    ListNode *p1, *p2;
    
    if (head == NULL || head->next == NULL) {
        return head;
    }

    p1 = p2 = head;

    while (1) {
        if (p2->next != NULL && p2->next->next != NULL) {
            p2 = p2->next->next;
            p1 = p1->next;
        } else {
            break;
        }
    }
    return p1;
}
```
- 思路分析：
  
  单链表的一个比较大的特点用一句广告语来说就是“不走回头路”，不能实现随机存取（random access）。如果我们想要找一个数组 `a` 的中间元素，直接 `a[len/2]` 就可以了，但是链表不行，因为只有 `a[len/2 - 1]` 知道 `a[len/2]` 在哪儿，其他人不知道。因此，如果按照数组的做法依样画葫芦，要找到链表的中点，我们需要做两步（1）知道链表有多长（2）从头结点开始顺序遍历到链表长度的一半的位置。这就需要 `1.5n`（ `n`为链表的长度）的时间复杂度了。有没有更好的办法呢？有的。想法很简单：两个人赛跑，如果 `A` 的速度是 `B` 的两倍的话，当 `A` 到终点的时候，`B` 应该刚到中点。这只需要遍历一遍链表就行了，还不用计算链表的长度。

  上面的代码就体现了这个想法。

## Q3：链表排序

```C++
double cmp(ListNode *p ,ListNode *q) {
    return (p->keyVal - q->keyVal);
}

ListNode* mergeSortList(ListNode *head) {
    ListNode *p, *q, *tail, *e;
    int nstep = 1;
    int nmerges = 0;
    int i;
    int psize, qsize;

    if (head == NULL || head->next == NULL) {
        return head;
    }

    while (1) {
        p = head;
        tail = NULL;
        nmerges = 0;
        while (p) {
            nmerges++;  q = p;  psize = 0;
            for (i = 0; i < nstep; i++) {
                psize++;
                q = q->next;
                if (q == NULL) {
                    break;
                }
            }
            qsize = nstep;
            while (psize >0 || (qsize >0 && q)) {
                if (psize == 0) {
                    e = q;
                    q = q->next;
                    qsize--;
                } elseif (q == NULL || qsize == 0) {
                    e = p;
                    p = p->next;
                    psize--;
                } elseif (cmp(p,q) <= 0) {
                    e = p; 
                    p = p->next;
                    psize--;
                } else {
                    e = q;
                    q = q->next;
                    qsize--;
                }
                if (tail != NULL) {
                    tail->next = e;
                } else {
                    head = e;
                }
                tail = e;
              }
            p = q;
        }
        tail->next = NULL;
        if (nmerges <= 1) {
            return head;
        } else {
            nstep <<= 1;
        }
    }
}
```

- 思路分析：

   链表排序最好使用归并排序算法。堆排序、快速排序这些在数组排序时性能非常好的算法，在链表只能“顺序访问”的魔咒下无法施展能力；但是归并排序却如鱼得水，非但保持了它 `O(nlogn)` 的时间复杂度，而且它在数组排序中广受诟病的空间复杂度在链表排序中也从 `O(n)` 降到了 `O(1)`。真是好得不得了啊，哈哈。以上程序是递推法的程序，另外值得一说的是看看那个时间复杂度，是不是有点眼熟？对！这就是分治法的时间复杂度，归并排序又是 `divide and conquer`。

## Q4：判断一个单链表是否有环

```C++
int is_looplist (ListNode *head) {
    ListNode *p1, *p2;
    p1 = p2 = head;

    if (head == NULL || head->next == NULL) {
        return 0;
    }

    while (p2->next != NULL && p2->next->next != NULL) {
        p1 = p1->next;
        p2 = p2->next->next;
        if (p1 == p2) {
            return 1;
        }
    }

    return 0;
}
```

- 思路分析

   这道题是《C专家编程》中的题了。其实算法也有很多，比如说：我觉得进行对访问过的结点进行标记这个想法也不错，而且在树遍历等场合我们也经常使用。但是在不允许做标记的场合就无法使用了。在种种限制的条件下，就有了上面的这种算法，其实思想很简单：就像两个人在操场上跑步一样，只要有个人的速度比另一个人的速度快一点，他们肯定会有相遇的时候的。不过带环链表与操场又不一样，带环链表的状态是离散的，所以选择走得快的要比走得慢的快多少很重要。比如说这里，如果一个指针一次走三步，一个指针一次走一步的话，很有可能它们虽然在一个环中但是永远遇不到，这要取决于环的大小以及两个指针初始位置相差多少了。你能看出两个指针的速度应该满足什么关系才能在有环的情况下相遇吗？

## 参考资料

1. 欧立奇等著《程序员面试宝典》，电子工业出版社
2. 《C专家编程》
3. Jurgen Appelo 《软件开发者面试百问》

*写于 2009 年 10 月*