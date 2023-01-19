#include "Setting.h"
#include "Random.h"
#include "Reader.h"
#include "Corrupt.h"
#include "Test.h"
#include <cstdlib>
#include <pthread.h>

extern "C"
void setInPath(char *path);

extern "C"
void setTrainPath(char *path);

extern "C"
void setValidPath(char *path);

extern "C"
void setTestPath(char *path);

extern "C"
void setEntPath(char *path);

extern "C"
void setRelPath(char *path);

extern "C"
void setOutPath(char *path);

extern "C"
void setWorkThreads(INT threads);

extern "C"
void setBern(INT con);

extern "C"
INT getWorkThreads();

extern "C"
INT getEntityTotal();

extern "C"
INT getRelationTotal();

extern "C"
INT getTripleTotal();

extern "C"
INT getTrainTotal();

extern "C"
INT getTestTotal();

extern "C"
INT getValidTotal();

extern "C"
void randReset();

extern "C"
void importTrainFiles();

struct Parameter {
	INT id;
	INT *batch_h;
	INT *batch_t;
	INT *batch_r;
	INT *batch_hb;
	INT *batch_tb;
	INT batchSize;
	INT negRate;
	INT negRelRate;
	bool p;
	bool val_loss;
	INT mode;
	bool filter_flag;
};

void* getBatch(void* con) {
	Parameter *para = (Parameter *)(con);
	INT id = para -> id;
	INT *batch_h = para -> batch_h;
	INT *batch_t = para -> batch_t;
	INT *batch_r = para -> batch_r;
	INT *batch_hb = para -> batch_hb;   // new
	INT *batch_tb = para -> batch_tb;   // new
	INT batchSize = para -> batchSize;
	INT negRate = para -> negRate;
	INT negRelRate = para -> negRelRate;

	bool p = para -> p;
	bool val_loss = para -> val_loss;
	INT mode = para -> mode;
	bool filter_flag = para -> filter_flag;

	INT lef, rig;   // 该线程处理这个batch的哪部分，lef-开始位置，rig-结束位置
	if (batchSize % workThreads == 0) {
		lef = id * (batchSize / workThreads);
		rig = (id + 1) * (batchSize / workThreads);
	} else {
		lef = id * (batchSize / workThreads + 1);
		rig = (id + 1) * (batchSize / workThreads + 1);
		if (rig > batchSize) rig = batchSize;
	}

	REAL prob = 500;
	if (val_loss == false) {
		for (INT batch = lef; batch < rig; batch++) {
			INT i = rand_max(id, trainTotal);
			batch_h[batch] = trainList[i].h;
			batch_t[batch] = trainList[i].t;
			batch_r[batch] = trainList[i].r;
			batch_hb[batch] = trainList[i].hb;
			batch_tb[batch] = trainList[i].tb;
			INT last = batchSize;   // 负样本在后面，offset
			for (INT times = 0; times < negRate; times ++) {    // negRate=25，每个正样本取25个负样本
				if (mode == 0){
				    // 生成相应的负样本
					if (bernFlag)
						prob = 1000 * right_mean[trainList[i].r] / (right_mean[trainList[i].r] + left_mean[trainList[i].r]);
					if (randd(id) % 1000 < prob) {
					    // 替换尾实体
						batch_h[batch + last] = trainList[i].h;
						batch_t[batch + last] = corrupt_head(id, trainList[i].h, trainList[i].r);
						batch_r[batch + last] = trainList[i].r;
						batch_hb[batch + last] = trainList[i].hb;
						batch_tb[batch + last] = trainList[i].tb;
					} else {
					    // 替换头实体
						batch_h[batch + last] = corrupt_tail(id, trainList[i].t, trainList[i].r);
						batch_t[batch + last] = trainList[i].t;
						batch_r[batch + last] = trainList[i].r;
						batch_hb[batch + last] = trainList[i].hb;
						batch_tb[batch + last] = trainList[i].tb;
					}
					last += batchSize;

				} else {
				    // 固定替换头或者尾
					if(mode == -1){
					    // 替换头实体
						batch_h[batch + last] = corrupt_tail(id, trainList[i].t, trainList[i].r);
						batch_t[batch + last] = trainList[i].t;
						batch_r[batch + last] = trainList[i].r;
					} else { // mode=1
					    // 替换尾实体
						batch_h[batch + last] = trainList[i].h;
						batch_t[batch + last] = corrupt_head(id, trainList[i].h, trainList[i].r);
						batch_r[batch + last] = trainList[i].r;
					}
					last += batchSize;
				}
			}
			for (INT times = 0; times < negRelRate; times++) {
				batch_h[batch + last] = trainList[i].h;
				batch_t[batch + last] = trainList[i].t;
				batch_r[batch + last] = corrupt_rel(id, trainList[i].h, trainList[i].t, trainList[i].r, p);
				last += batchSize;
			}
		}
	}
	else
	{
		for (INT batch = lef; batch < rig; batch++)
		{
			batch_h[batch] = validList[batch].h;
			batch_t[batch] = validList[batch].t;
			batch_r[batch] = validList[batch].r;
		}
	}
	pthread_exit(NULL);
}

extern "C"
void sampling(
		INT *batch_h, 
		INT *batch_t, 
		INT *batch_r,
		INT *batch_hb,  // new
		INT *batch_tb,  // new
		INT batchSize, 
		INT negRate = 1, 
		INT negRelRate = 0, 
		INT mode = 0,
		bool filter_flag = true,
		bool p = false, 
		bool val_loss = false
) {
	pthread_t *pt = (pthread_t *)malloc(workThreads * sizeof(pthread_t));
	Parameter *para = (Parameter *)malloc(workThreads * sizeof(Parameter));
	for (INT threads = 0; threads < workThreads; threads++) {
	    // 给每个线程的参数赋值
		para[threads].id = threads;
		para[threads].batch_h = batch_h;
		para[threads].batch_t = batch_t;
		para[threads].batch_r = batch_r;
		para[threads].batch_hb = batch_hb;
		para[threads].batch_tb = batch_tb;
		para[threads].batchSize = batchSize;
		para[threads].negRate = negRate;
		para[threads].negRelRate = negRelRate;
		para[threads].p = p;
		para[threads].val_loss = val_loss;
		para[threads].mode = mode;
		para[threads].filter_flag = filter_flag;
		// 创建线程，线程执行的函数getBatch，函数参数为para[threads]
		pthread_create(&pt[threads], NULL, getBatch, (void*)(para+threads));
	}
	for (INT threads = 0; threads < workThreads; threads++)
		pthread_join(pt[threads], NULL);

	free(pt);
	free(para);
}

int main() {
	importTrainFiles();
	return 0;
}