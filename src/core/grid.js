(function(root){
	"use strict";
	const Fluid=root.Fluid=root.Fluid||{};
	Fluid.core=Fluid.core||{};
	function idx(x,y,texW){
		return x+y*texW;
	}
	function isInterior(x,y,texW,texH){
		return x>=1&&x<texW-1&&y>=1&&y<texH-1;
	}
	function texelCount(texW,texH){
		return texW*texH;
	}
	function bytesPerRowForFormat(texW,bytesPerCell){
		let row=texW*bytesPerCell;
		return Math.ceil(row/256)*256;
	}
	function createField(texW,texH,bytesPerCell,Kind){
		let n=texW*texH;
		return new Kind(n*bytesPerCell/Kind.BYTES_PER_ELEMENT);
	}
	function fillField(a,value){
		for(let i=0;i<a.length;i++)a[i]=value;
	}
	function allocPingPong(texW,texH,bytesPerCell,Kind){
		let a=createField(texW,texH,bytesPerCell,Kind);
		let b=createField(texW,texH,bytesPerCell,Kind);
		return {a,b,cur:0};
	}
	function curBuf(pp){
		return pp.cur===0?pp.a:pp.b;
	}
	function nxtBuf(pp){
		return pp.cur===0?pp.b:pp.a;
	}
	function swap(pp){
		pp.cur=1-pp.cur;
	}
	Fluid.core.grid={idx,isInterior,texelCount,bytesPerRowForFormat,createField,fillField,allocPingPong,curBuf,nxtBuf,swap};
	if(typeof module!=="undefined"&&module.exports)module.exports=Fluid.core.grid;
})(globalThis);
