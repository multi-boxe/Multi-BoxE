#ifndef TRIPLE_H
#define TRIPLE_H
#include "Setting.h"

struct Triple {

	INT h, r, t;
	INT hb, tb;

	static bool cmp_head(const Triple &a, const Triple &b) {
		return (a.h < b.h)||(a.h == b.h && a.r < b.r)||(a.h == b.h && a.r == b.r && a.t < b.t);
	}

	static bool cmp_tail(const Triple &a, const Triple &b) {
		return (a.t < b.t)||(a.t == b.t && a.r < b.r)||(a.t == b.t && a.r == b.r && a.h < b.h);
	}

	static bool cmp_rel(const Triple &a, const Triple &b) {
		return (a.h < b.h)||(a.h == b.h && a.t < b.t)||(a.h == b.h && a.t == b.t && a.r < b.r);
	}

	static bool cmp_rel2(const Triple &a, const Triple &b) {
		return (a.r < b.r)||(a.r == b.r && a.h < b.h)||(a.r == b.r && a.h == b.h && a.t < b.t);
	}

};

struct Boxes {
    INT hb;
    INT tb;
};

#endif
