"use strict";
// Headless contract check for the view-mode row (src/app/viewmenu.js) and the
// keyboard route through it (src/app/input.js). A tiny DOM stub is enough: no
// browser, no dependencies, no shaders. Run: node experiments/ui.js
let checks=0,fails=0;
function ok(cond,msg){
	checks++;
	if(cond)return console.log("ok   "+msg);
	fails++;
	console.log("FAIL "+msg);
}

class El{
	constructor(tag){
		this.tagName=tag;
		this.children=[];
		this.parentNode=null;
		this.listeners=Object.create(null);
		this.type="";
		this.name="";
		this.value="";
		this._checked=false;
		this.textContent="";
	}
	// native radio behavior: checking one unchecks its group
	get checked(){return this._checked;}
	set checked(v){
		this._checked=!!v;
		if(!this._checked)return;
		let root=this;
		while(root.parentNode)root=root.parentNode;
		const self=this;
		(function walk(n){
			for(const c of n.children){
				if(c!==self&&c.tagName==="INPUT"&&c.type===self.type&&c.name===self.name)c._checked=false;
				walk(c);
			}
		})(root);
	}
	// label.control: the wrapped input (the label has exactly one)
	get control(){
		for(const c of this.children)if(c.tagName==="INPUT")return c;
		return null;
	}
	appendChild(c){
		c.parentNode=this;
		this.children.push(c);
		return c;
	}
	addEventListener(type,fn){
		(this.listeners[type]||(this.listeners[type]=[])).push(fn);
	}
	dispatch(type,target){
		for(let node=target;node;node=node.parentNode){
			const ls=node.listeners[type];
			if(!ls)continue;
			const ev={type,target,preventDefault(){}};
			for(const fn of ls)fn.call(node,ev);
		}
	}
}
function textNode(text){
	return {nodeType:3,textContent:text,parentNode:null,tagName:"",children:[],listeners:Object.create(null)};
}
const container=new El("DIV");
container.id="view-menu";
const windowListeners=Object.create(null);
const mq={matches:true};
globalThis.document={
	readyState:"complete",
	getElementById:id=>id==="view-menu"?container:null,
	createElement:tag=>new El(tag.toUpperCase()),
	createTextNode:textNode
};
globalThis.matchMedia=()=>({get matches(){return mq.matches;}});
globalThis.window=globalThis;
globalThis.addEventListener=(type,fn)=>{(windowListeners[type]||(windowListeners[type]=[])).push(fn);};
function key(k){
	for(const fn of windowListeners.keydown||[])fn({key:k,preventDefault(){}});
}

require("../src/core/config.js");
require("../src/app/viewmenu.js");
require("../src/app/input.js");

const cfg=Object.assign({},Fluid.core.config.defaults);
cfg._dirty=false;
const menu=Fluid.app.viewMenu.create(cfg);
const labels=container.children;
const radios=labels.map(l=>l.control);
const checkedCount=()=>radios.filter(r=>r.checked).length;

ok(labels.length===8,"one label per mode");
ok(labels.every(l=>l.control&&l.control.type==="radio"),"every label wraps a radio");
ok(radios.every((r,i)=>r.name==="view-mode"&&r.value===String(i)),"group name and values");
ok(checkedCount()===1&&radios[0].checked,"exactly one checked at boot");
ok(labels.every(l=>l.children[1].nodeType===3&&l.children[1].textContent.length>0),"label text follows the input");

// click: the browser checks the radio, then bubbles change to the container
cfg._dirty=false;
radios[2].checked=true;
container.dispatch("change",radios[2]);
ok(cfg.viewMode===2,"change sets the mode");
ok(cfg._dirty===true,"change marks the view dirty");
ok(checkedCount()===1&&radios[2].checked,"one checked after click");

// single writer: an already active id is a no-op, unknown ids are ignored
cfg._dirty=false;
menu.setMode(2);
ok(cfg._dirty===false,"setMode on the active mode returns before dirtying");
menu.setMode(99);
menu.setMode(-1);
ok(cfg.viewMode===2&&cfg._dirty===false,"setMode ignores unknown ids");

// keyboard 0-7 must go through the same writer and sync the radio
Fluid.app.input.create({addEventListener(){},getBoundingClientRect(){return {left:0,top:0,width:1,height:1};}},cfg,null,menu.setMode);
key("5");
ok(cfg.viewMode===5,"key 5 routes through setMode");
ok(radios[5].checked&&checkedCount()===1,"keyboard checks the radio");

// hover is gated on a fine pointer that can hover
mq.matches=false;
menu.setMode(2);
container.dispatch("pointerover",radios[3]);
ok(cfg.viewMode===2,"touch/stylus pointerover does not switch");
ok(radios[2].checked,"touch/stylus leaves the checked radio alone");

mq.matches=true;
container.dispatch("pointerover",radios[3]);
ok(cfg.viewMode===3,"fine-pointer hover switches");
ok(radios[3].checked&&checkedCount()===1,"hover checks exactly one radio");

// bubbles from the label text; the walk stops at the container for non-labels
container.dispatch("pointerover",labels[6].children[1]);
ok(cfg.viewMode===6,"hover from the label text finds the radio");
container.dispatch("pointerover",container);
ok(cfg.viewMode===6,"pointerover outside a label does not switch");

// a change event from an unrelated control is not a view change
cfg._dirty=false;
const other=new El("INPUT");
other.type="checkbox";
other.name="nope";
other.value="9";
container.dispatch("change",other);
ok(cfg.viewMode===6&&cfg._dirty===false,"change from a foreign input is ignored");

console.log(fails?"\n"+fails+" FAILED of "+checks:"\nall "+checks+" checks pass");
process.exit(fails?1:0);
