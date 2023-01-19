#ifndef TEST_H
#define TEST_H
#include "Setting.h"
#include "Reader.h"
#include "Corrupt.h"

/*=====================================================================================
link prediction
======================================================================================*/
INT lastHead = 0;
INT lastTail = 0;
INT lastRel = 0;
REAL l1_filter_tot = 0, l1_tot = 0, r1_tot = 0, r1_filter_tot = 0, l_tot = 0, r_tot = 0, l_filter_rank = 0, l_rank = 0, l_filter_reci_rank = 0, l_reci_rank = 0;
REAL l3_filter_tot = 0, l3_tot = 0, r3_tot = 0, r3_filter_tot = 0, l_filter_tot = 0, r_filter_tot = 0, r_filter_rank = 0, r_rank = 0, r_filter_reci_rank = 0, r_reci_rank = 0;
REAL rel3_tot = 0, rel3_filter_tot = 0, rel_filter_tot = 0, rel_filter_rank = 0, rel_rank = 0, rel_filter_reci_rank = 0, rel_reci_rank = 0, rel_tot = 0, rel1_tot = 0, rel1_filter_tot = 0;

REAL l1_filter_tot_constrain = 0, l1_tot_constrain = 0, r1_tot_constrain = 0, r1_filter_tot_constrain = 0, l_tot_constrain = 0, r_tot_constrain = 0, l_filter_rank_constrain = 0, l_rank_constrain = 0, l_filter_reci_rank_constrain = 0, l_reci_rank_constrain = 0;
REAL l3_filter_tot_constrain = 0, l3_tot_constrain = 0, r3_tot_constrain = 0, r3_filter_tot_constrain = 0, l_filter_tot_constrain = 0, r_filter_tot_constrain = 0, r_filter_rank_constrain = 0, r_rank_constrain = 0, r_filter_reci_rank_constrain = 0, r_reci_rank_constrain = 0;
REAL hit1, hit3, hit10, mr, mrr;
REAL hit1TC, hit3TC, hit10TC, mrTC, mrrTC;

extern "C"
void initTest() {
    lastHead = 0;   // 只是用于确定具体哪个测试样本的
    lastTail = 0;
    lastRel = 0;
    l1_filter_tot = 0, l1_tot = 0, r1_tot = 0, r1_filter_tot = 0, l_tot = 0, r_tot = 0, l_filter_rank = 0, l_rank = 0, l_filter_reci_rank = 0, l_reci_rank = 0;
    l3_filter_tot = 0, l3_tot = 0, r3_tot = 0, r3_filter_tot = 0, l_filter_tot = 0, r_filter_tot = 0, r_filter_rank = 0, r_rank = 0, r_filter_reci_rank = 0, r_reci_rank = 0;
    REAL rel3_tot = 0, rel3_filter_tot = 0, rel_filter_tot = 0, rel_filter_rank = 0, rel_rank = 0, rel_filter_reci_rank = 0, rel_reci_rank = 0, rel_tot = 0, rel1_tot = 0, rel1_filter_tot = 0;

    l1_filter_tot_constrain = 0, l1_tot_constrain = 0, r1_tot_constrain = 0, r1_filter_tot_constrain = 0, l_tot_constrain = 0, r_tot_constrain = 0, l_filter_rank_constrain = 0, l_rank_constrain = 0, l_filter_reci_rank_constrain = 0, l_reci_rank_constrain = 0;
    l3_filter_tot_constrain = 0, l3_tot_constrain = 0, r3_tot_constrain = 0, r3_filter_tot_constrain = 0, l_filter_tot_constrain = 0, r_filter_tot_constrain = 0, r_filter_rank_constrain = 0, r_rank_constrain = 0, r_filter_reci_rank_constrain = 0, r_reci_rank_constrain = 0;
}

extern "C"
void getHeadBatch(INT *ph, INT *pt, INT *pr) {
    // *ph等就是numpy数组所在的内存地址
    // 替换头实体，在此将测试集中的triple赋值给numpy数组
    // ph:pt:pr -> 0~entityTotal-1 : lastHead.t : lastHead.r
    for (INT i = 0; i < entityTotal; i++) {
        ph[i] = i;
        pt[i] = testList[lastHead].t;
        pr[i] = testList[lastHead].r;
    }
    lastHead++;
}

extern "C"
void getTailBatch(INT *ph, INT *pt, INT *pr) {
    for (INT i = 0; i < entityTotal; i++) {
        ph[i] = testList[lastTail].h;
        pt[i] = i;
        pr[i] = testList[lastTail].r;
    }
    lastTail++;
}

extern "C"
void getRelBatch(INT *ph, INT *pt, INT *pr) {
    for (INT i = 0; i < relationTotal; i++) {
        ph[i] = testList[lastRel].h;
        pt[i] = testList[lastRel].t;
        pr[i] = i;
    }
}

//==========================================
// 包含box信息的
extern "C"
void getHeadBatch2(INT *ph, INT *pt, INT *pr, INT *phb, INT *ptb, INT *hid, INT *tid) {
    pr[0] = testList[lastHead].r;
    pt[0] = testList[lastHead].t;
    phb[0] = testList[lastHead].hb;
    ptb[0] = testList[lastHead].tb;

    hid[0] = testList[lastHead].h;
    tid[0] = testList[lastHead].t;

    for (INT i = 0; i < entityTotal; i++) {
        ph[i] = i;
    }
    lastHead++;
}

extern "C"
void getTailBatch2(INT *ph, INT *pt, INT *pr, INT *phb, INT *ptb, INT *hid, INT *tid) {
    ph[0] = testList[lastTail].h;
    pr[0] = testList[lastTail].r;
    phb[0] = testList[lastTail].hb;
    ptb[0] = testList[lastTail].tb;

    hid[0] = testList[lastTail].h;
    tid[0] = testList[lastTail].t;

    for (INT i = 0; i < entityTotal; i++) {
        pt[i] = i;
    }
    lastTail++;
}
//==========================================

extern "C"
INT testHead(REAL *con, INT lastHead, bool type_constrain = false) {
    // 对一个测试样例的所有score进行筛选统计
    // lastHead 当前测试样例的index
    INT h = testList[lastHead].h;
    INT t = testList[lastHead].t;
    INT r = testList[lastHead].r;
    INT lef, rig;
    if (type_constrain) {
        lef = head_lef[r];
        rig = head_rig[r];
    }
    REAL minimal = con[h];  // 把测试样例对应的score作为当前最小score
    INT l_s = 0;
    INT l_filter_s = 0;
    INT l_s_constrain = 0;
    INT l_filter_s_constrain = 0;

    for (INT j = 0; j < entityTotal; j++) {
        if (j != h) {
            REAL value = con[j];
            if (value < minimal) {
                l_s += 1;
                if (not _find(j, t, r))
                    l_filter_s += 1;
            }
            if (type_constrain) {
                while (lef < rig && head_type[lef] < j) lef ++;
                if (lef < rig && j == head_type[lef]) {
                    if (value < minimal) {
                        l_s_constrain += 1;
                        if (not _find(j, t, r)) {
                            l_filter_s_constrain += 1;
                        }
                    }  
                }
            }
        }
    }

    if (l_filter_s < 10) l_filter_tot += 1;
    if (l_s < 10) l_tot += 1;
    if (l_filter_s < 3) l3_filter_tot += 1;
    if (l_s < 3) l3_tot += 1;
    if (l_filter_s < 1) l1_filter_tot += 1;
    if (l_s < 1) l1_tot += 1;

    l_filter_rank += (l_filter_s+1);
    l_rank += (1 + l_s);
    l_filter_reci_rank += 1.0/(l_filter_s+1);
    l_reci_rank += 1.0/(l_s+1);

    if (type_constrain) {
        if (l_filter_s_constrain < 10) l_filter_tot_constrain += 1;
        if (l_s_constrain < 10) l_tot_constrain += 1;
        if (l_filter_s_constrain < 3) l3_filter_tot_constrain += 1;
        if (l_s_constrain < 3) l3_tot_constrain += 1;
        if (l_filter_s_constrain < 1) l1_filter_tot_constrain += 1;
        if (l_s_constrain < 1) l1_tot_constrain += 1;

        l_filter_rank_constrain += (l_filter_s_constrain+1);
        l_rank_constrain += (1+l_s_constrain);
        l_filter_reci_rank_constrain += 1.0/(l_filter_s_constrain+1);
        l_reci_rank_constrain += 1.0/(l_s_constrain+1);
    }

    return ++l_filter_s;
}

extern "C"
INT testTail(REAL *con, INT lastTail, bool type_constrain = false) {
    INT h = testList[lastTail].h;
    INT t = testList[lastTail].t;
    INT r = testList[lastTail].r;
    INT lef, rig;
    if (type_constrain) {
        lef = tail_lef[r];
        rig = tail_rig[r];
    }
    REAL minimal = con[t];
    INT r_s = 0;
    INT r_filter_s = 0;
    INT r_s_constrain = 0;
    INT r_filter_s_constrain = 0;
    for (INT j = 0; j < entityTotal; j++) {
        if (j != t) {
            REAL value = con[j];
            if (value < minimal) {
                r_s += 1;
                if (not _find(h, j, r))
                    r_filter_s += 1;
            }
            if (type_constrain) {
                while (lef < rig && tail_type[lef] < j) lef ++;
                if (lef < rig && j == tail_type[lef]) {
                        if (value < minimal) {
                            r_s_constrain += 1;
                            if (not _find(h, j ,r)) {
                                r_filter_s_constrain += 1;
                            }
                        }
                }
            }
        }
        
    }

    if (r_filter_s < 10) r_filter_tot += 1;
    if (r_s < 10) r_tot += 1;
    if (r_filter_s < 3) r3_filter_tot += 1;
    if (r_s < 3) r3_tot += 1;
    if (r_filter_s < 1) r1_filter_tot += 1;
    if (r_s < 1) r1_tot += 1;

    r_filter_rank += (1+r_filter_s);
    r_rank += (1+r_s);
    r_filter_reci_rank += 1.0/(1+r_filter_s);
    r_reci_rank += 1.0/(1+r_s);
    
    if (type_constrain) {
        if (r_filter_s_constrain < 10) r_filter_tot_constrain += 1;
        if (r_s_constrain < 10) r_tot_constrain += 1;
        if (r_filter_s_constrain < 3) r3_filter_tot_constrain += 1;
        if (r_s_constrain < 3) r3_tot_constrain += 1;
        if (r_filter_s_constrain < 1) r1_filter_tot_constrain += 1;
        if (r_s_constrain < 1) r1_tot_constrain += 1;

        r_filter_rank_constrain += (1+r_filter_s_constrain);
        r_rank_constrain += (1+r_s_constrain);
        r_filter_reci_rank_constrain += 1.0/(1+r_filter_s_constrain);
        r_reci_rank_constrain += 1.0/(1+r_s_constrain);
    }

    return ++r_filter_s;
}

extern "C"
void testRel(REAL *con) {
    INT h = testList[lastRel].h;
    INT t = testList[lastRel].t;
    INT r = testList[lastRel].r;

    REAL minimal = con[r];
    INT rel_s = 0;
    INT rel_filter_s = 0;

    for (INT j = 0; j < relationTotal; j++) {
        if (j != r) {
            REAL value = con[j];
            if (value < minimal) {
                rel_s += 1;
                if (not _find(h, t, j))
                    rel_filter_s += 1;
            }
        }
    }

    if (rel_filter_s < 10) rel_filter_tot += 1;
    if (rel_s < 10) rel_tot += 1;
    if (rel_filter_s < 3) rel3_filter_tot += 1;
    if (rel_s < 3) rel3_tot += 1;
    if (rel_filter_s < 1) rel1_filter_tot += 1;
    if (rel_s < 1) rel1_tot += 1;

    rel_filter_rank += (rel_filter_s+1);
    rel_rank += (1+rel_s);
    rel_filter_reci_rank += 1.0/(rel_filter_s+1);
    rel_reci_rank += 1.0/(rel_s+1);

    lastRel++;
}


extern "C"
void test_link_prediction(bool type_constrain = false) {
    l_rank /= testTotal;
    r_rank /= testTotal;
    l_reci_rank /= testTotal;
    r_reci_rank /= testTotal;
 
    l_tot /= testTotal;
    l3_tot /= testTotal;
    l1_tot /= testTotal;
 
    r_tot /= testTotal;
    r3_tot /= testTotal;
    r1_tot /= testTotal;

    // with filter
    l_filter_rank /= testTotal;
    r_filter_rank /= testTotal;
    l_filter_reci_rank /= testTotal;
    r_filter_reci_rank /= testTotal;
 
    l_filter_tot /= testTotal;
    l3_filter_tot /= testTotal;
    l1_filter_tot /= testTotal;
 
    r_filter_tot /= testTotal;
    r3_filter_tot /= testTotal;
    r1_filter_tot /= testTotal;

    printf("no type constraint results:\n");
    
    printf("metric:\t\t\t MRR \t\t MR \t\t hit@10 \t hit@3  \t hit@1 \n");
    printf("l(raw):\t\t\t %f \t %f \t %f \t %f \t %f \n", l_reci_rank, l_rank, l_tot, l3_tot, l1_tot);
    printf("r(raw):\t\t\t %f \t %f \t %f \t %f \t %f \n", r_reci_rank, r_rank, r_tot, r3_tot, r1_tot);
    printf("averaged(raw):\t\t %f \t %f \t %f \t %f \t %f \n",
            (l_reci_rank+r_reci_rank)/2, (l_rank+r_rank)/2, (l_tot+r_tot)/2, (l3_tot+r3_tot)/2, (l1_tot+r1_tot)/2);
    printf("\n");
    printf("l(filter):\t\t %f \t %f \t %f \t %f \t %f \n", l_filter_reci_rank, l_filter_rank, l_filter_tot, l3_filter_tot, l1_filter_tot);
    printf("r(filter):\t\t %f \t %f \t %f \t %f \t %f \n", r_filter_reci_rank, r_filter_rank, r_filter_tot, r3_filter_tot, r1_filter_tot);
    printf("averaged(filter):\t %f \t %f \t %f \t %f \t %f \n",
            (l_filter_reci_rank+r_filter_reci_rank)/2, (l_filter_rank+r_filter_rank)/2, (l_filter_tot+r_filter_tot)/2, (l3_filter_tot+r3_filter_tot)/2, (l1_filter_tot+r1_filter_tot)/2);

    mrr = (l_filter_reci_rank+r_filter_reci_rank) / 2;
    mr = (l_filter_rank+r_filter_rank) / 2;
    hit10 = (l_filter_tot+r_filter_tot) / 2;
    hit3 = (l3_filter_tot+r3_filter_tot) / 2;
    hit1 = (l1_filter_tot+r1_filter_tot) / 2;

    if (type_constrain) {
        //type constrain
        l_rank_constrain /= testTotal;
        r_rank_constrain /= testTotal;
        l_reci_rank_constrain /= testTotal;
        r_reci_rank_constrain /= testTotal;
     
        l_tot_constrain /= testTotal;
        l3_tot_constrain /= testTotal;
        l1_tot_constrain /= testTotal;
     
        r_tot_constrain /= testTotal;
        r3_tot_constrain /= testTotal;
        r1_tot_constrain /= testTotal;

        // with filter
        l_filter_rank_constrain /= testTotal;
        r_filter_rank_constrain /= testTotal;
        l_filter_reci_rank_constrain /= testTotal;
        r_filter_reci_rank_constrain /= testTotal;
     
        l_filter_tot_constrain /= testTotal;
        l3_filter_tot_constrain /= testTotal;
        l1_filter_tot_constrain /= testTotal;
     
        r_filter_tot_constrain /= testTotal;
        r3_filter_tot_constrain /= testTotal;
        r1_filter_tot_constrain /= testTotal;

        printf("type constraint results:\n");
        
        printf("metric:\t\t\t MRR \t\t MR \t\t hit@10 \t hit@3  \t hit@1 \n");
        printf("l(raw):\t\t\t %f \t %f \t %f \t %f \t %f \n", l_reci_rank_constrain, l_rank_constrain, l_tot_constrain, l3_tot_constrain, l1_tot_constrain);
        printf("r(raw):\t\t\t %f \t %f \t %f \t %f \t %f \n", r_reci_rank_constrain, r_rank_constrain, r_tot_constrain, r3_tot_constrain, r1_tot_constrain);
        printf("averaged(raw):\t\t %f \t %f \t %f \t %f \t %f \n",
                (l_reci_rank_constrain+r_reci_rank_constrain)/2, (l_rank_constrain+r_rank_constrain)/2, (l_tot_constrain+r_tot_constrain)/2, (l3_tot_constrain+r3_tot_constrain)/2, (l1_tot_constrain+r1_tot_constrain)/2);
        printf("\n");
        printf("l(filter):\t\t %f \t %f \t %f \t %f \t %f \n", l_filter_reci_rank_constrain, l_filter_rank_constrain, l_filter_tot_constrain, l3_filter_tot_constrain, l1_filter_tot_constrain);
        printf("r(filter):\t\t %f \t %f \t %f \t %f \t %f \n", r_filter_reci_rank_constrain, r_filter_rank_constrain, r_filter_tot_constrain, r3_filter_tot_constrain, r1_filter_tot_constrain);
        printf("averaged(filter):\t %f \t %f \t %f \t %f \t %f \n",
                (l_filter_reci_rank_constrain+r_filter_reci_rank_constrain)/2, (l_filter_rank_constrain+r_filter_rank_constrain)/2, (l_filter_tot_constrain+r_filter_tot_constrain)/2, (l3_filter_tot_constrain+r3_filter_tot_constrain)/2, (l1_filter_tot_constrain+r1_filter_tot_constrain)/2);

        mrrTC = (l_filter_reci_rank_constrain+r_filter_reci_rank_constrain)/2;
        mrTC = (l_filter_rank_constrain+r_filter_rank_constrain) / 2;
        hit10TC = (l_filter_tot_constrain+r_filter_tot_constrain) / 2;
        hit3TC = (l3_filter_tot_constrain+r3_filter_tot_constrain) / 2;
        hit1TC = (l1_filter_tot_constrain+r1_filter_tot_constrain) / 2;
    }
}

extern "C"
void test_relation_prediction() {
    rel_rank /= testTotal;
    rel_reci_rank /= testTotal;
  
    rel_tot /= testTotal;
    rel3_tot /= testTotal;
    rel1_tot /= testTotal;

    // with filter
    rel_filter_rank /= testTotal;
    rel_filter_reci_rank /= testTotal;
  
    rel_filter_tot /= testTotal;
    rel3_filter_tot /= testTotal;
    rel1_filter_tot /= testTotal;

    printf("no type constraint results:\n");
    
    printf("metric:\t\t\t MRR \t\t MR \t\t hit@10 \t hit@3  \t hit@1 \n");
    printf("averaged(raw):\t\t %f \t %f \t %f \t %f \t %f \n",
            rel_reci_rank, rel_rank, rel_tot, rel3_tot, rel1_tot);
    printf("\n");
    printf("averaged(filter):\t %f \t %f \t %f \t %f \t %f \n",
            rel_filter_reci_rank, rel_filter_rank, rel_filter_tot, rel3_filter_tot, rel1_filter_tot);
}

extern "C"
REAL getTestLinkHit10(bool type_constrain = false) {
    if (type_constrain)
        return hit10TC;
    printf("%f\n", hit10);
    return hit10;
}

extern "C"
REAL  getTestLinkHit3(bool type_constrain = false) {
    if (type_constrain)
        return hit3TC;
    return hit3;
}

extern "C"
REAL  getTestLinkHit1(bool type_constrain = false) {
    if (type_constrain)
        return hit1TC;    
    return hit1;
}

extern "C"
REAL  getTestLinkMR(bool type_constrain = false) {
    if (type_constrain)
        return mrTC;
    return mr;
}

extern "C"
REAL  getTestLinkMRR(bool type_constrain = false) {
    if (type_constrain)
        return mrrTC;    
    return mrr;
}

/*=====================================================================================
MyTester —— 获得每个样例的filter
======================================================================================*/
extern "C"
void getHMask(REAL *con, INT h, INT t, INT r, INT *res) {
    // 替换头实体的mask
    INT i = 0;

    REAL current = con[h];

    for (INT j = 0; j < entityTotal; j++) {
        if (j != h) {   // 非当前triple
            if(con[j] < current){ // 排在当前样例之前
                if (not _find(j, t, r)) // filter
                    res[i++] = j;
            }
        }else{
            res[i++] = h;
        }
    }
}

extern "C"
void getTMask(REAL *con, INT h, INT t, INT r, INT *res) {
    // 替换尾实体的mask
    INT i = 0;

    REAL current = con[t];

    for (INT j = 0; j < entityTotal; j++) {
        if (j != t) {   // 非当前triple
            if(con[j] < current){ // 排在当前样例之前
                if (not _find(h, j, r)) // filter
                    res[i++] = j;
            }
        }else{
            res[i++] = t;
        }
    }
}

/*=====================================================================================
triple classification
======================================================================================*/
Triple *negTestList = NULL;

extern "C"
void getNegTest() {
    if (negTestList == NULL)
        negTestList = (Triple *)calloc(testTotal, sizeof(Triple));
    for (INT i = 0; i < testTotal; i++) {
        negTestList[i] = testList[i];
        if (randd(0) % 1000 < 500)
            negTestList[i].t = corrupt_head(0, testList[i].h, testList[i].r);
        else
            negTestList[i].h = corrupt_tail(0, testList[i].t, testList[i].r);
    }
}

extern "C"
void getTestBatch(INT *ph, INT *pt, INT *pr, INT *nh, INT *nt, INT *nr) {
    getNegTest();
    for (INT i = 0; i < testTotal; i++) {
        ph[i] = testList[i].h;
        pt[i] = testList[i].t;
        pr[i] = testList[i].r;
        nh[i] = negTestList[i].h;
        nt[i] = negTestList[i].t;
        nr[i] = negTestList[i].r;
    }
}
#endif
