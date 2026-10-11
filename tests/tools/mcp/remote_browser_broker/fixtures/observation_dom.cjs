const readline = require('node:readline');
const vm = require('node:vm');
// Synthetic DOM/CDP fixture. It executes production expressions; it is not a browser conformance test.
const contexts = new Map();
let document;
let location = {href:'http://127.0.0.1/', get origin() {return new URL(this.href).origin;}, reload() {reset('<main>Reloaded</main>', this.href);}};
const decode = x => x.replace(/&quot;/g,'"').replace(/&lt;/g,'<').replace(/&gt;/g,'>').replace(/&amp;/g,'&');
const selectors = x => x.split(/,(?![^\[]*\])/).map(s=>s.trim());
function matches(el, selector) {
  return selectors(selector).some(s => {
    const parts = s.split(/\s+(?![^\[]*\])/);
    if (parts.length > 1) {
      const last = parts.pop();
      return matches(el,last) && !!el.parentElement?.closest(parts.join(' '));
    }
    if (s === '*') return true;
    const tag = s.match(/^[a-zA-Z][\w-]*/);
    if (tag && el.tagName !== tag[0].toUpperCase()) return false;
    const id = s.match(/#([\w-]+)/);
    if (id && el.id !== id[1]) return false;
    const attrs = [...s.matchAll(/\[([\w-]+)(?:(\^?=)["']?([^\]"']*)["']?)?\]/g)];
    for (const [,key,op,value] of attrs) {
      const got = el.getAttribute(key);
      if (got === null || (op === '=' && got !== value) || (op === '^=' && !got.startsWith(value))) return false;
    }
    if(!tag && !id && !attrs.length) throw new Error('Invalid fixture selector');
    return true;
  });
}
class TextNode {
  constructor(text) {this.nodeType=3; this.textContent=text; this.parentElement=null;}
}
class Element {
  constructor(tag, attrs={}) {
    this.nodeType=1; this.tagName=tag.toUpperCase(); this.attrs=attrs; this.childNodes=[];
    this.parentElement=null; this.ownerDocument=document; this.dataset=new Proxy({}, {set(target,key,value) {target[key]=String(value);return true;}}); this.clicks=0;
  }
  get id() {return this.attrs.id || '';}
  get hidden() {return 'hidden' in this.attrs;}
  get type() {return this.attrs.type || '';}
  get isConnected() {return this===document.body || !!this.parentElement?.isConnected;}
  get innerText() {return this.hidden ? '' : this.childNodes.map(x=>x.nodeType===3 ? x.textContent : x.innerText).join('');}
  set innerText(text) {this.textContent=text;}
  get textContent() {return this.childNodes.map(x=>x.textContent).join('');}
  set textContent(text) {this.childNodes=[]; this.append(new TextNode(String(text)));}
  get innerHTML() {return this.childNodes.map(serialize).join('');}
  set innerHTML(html) {this.childNodes=[]; parse(String(html),this);}
  get outerHTML() {return serialize(this);}
  getAttribute(name) {return name in this.attrs ? this.attrs[name] : null;}
  setAttribute(name,value) {this.attrs[name]=String(value);}
  getClientRects() {return this.hidden || this.attrs.style?.includes('display:none') || (this.parentElement && !this.parentElement.getClientRects().length) ? [] : [{}];}
  matches(selector) {return matches(this,selector);}
  contains(el) {return this === el || this.childNodes.some(x=>x.nodeType===1 && x.contains(el));}
  querySelectorAll(selector) {return flatten(this).slice(1).filter(el=>matches(el,selector));}
  querySelector(selector) {return this.querySelectorAll(selector)[0] || null;}
  closest(selector) {return matches(this,selector) ? this : this.parentElement?.closest(selector) || null;}
  compareDocumentPosition(other) {return flatten(document.body).indexOf(other)>flatten(document.body).indexOf(this) ? 4 : 2;}
  append(node) {node.parentElement=this; this.childNodes.push(node);}
  remove() {if(this.parentElement) this.parentElement.childNodes=this.parentElement.childNodes.filter(x=>x!==this); this.parentElement=null;}
  insertAdjacentHTML(where,html) {const temp=new Element('div'); parse(html,temp); const children=temp.childNodes; children.forEach(x=>x.parentElement=this); this.childNodes=where==='afterbegin' ? children.concat(this.childNodes) : this.childNodes.concat(children);}
  click() {this.clicks++; if(this.attrs.onclick) new Function(this.attrs.onclick).call(this);}
  focus() {} scrollIntoView() {} dispatchEvent() {}
}
class HTMLAnchorElement extends Element {get href() {return new URL(this.attrs.href||'',location.href).href;}}
function flatten(el) {return [el,...el.childNodes.filter(x=>x.nodeType===1).flatMap(flatten)];}
function serialize(node) {return node.nodeType===3 ? node.textContent : '<'+node.tagName.toLowerCase()+Object.entries(node.attrs).map(([k,v])=>' '+k+'="'+v+'"').join('')+'>'+node.childNodes.map(serialize).join('')+'</'+node.tagName.toLowerCase()+'>';}
function parse(html,root) {
  const stack=[root];
  for(const token of html.match(/<[^>]+>|[^<]+/g)||[]) {
    if(token.startsWith('</')) {if(stack.length>1) stack.pop(); continue;}
    if(token.startsWith('<!')) continue;
    if(token.startsWith('<')) {
      const tag=token.match(/^<([\w-]+)/)?.[1]; if(!tag) continue;
      const attrs={};
      for(const [,key,dq,sq,bare] of token.slice(tag.length+1,-1).matchAll(/([\w-]+)(?:\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s>]+)))?/g)) attrs[key]=decode(dq??sq??bare??'');
      const node=tag==='a' ? new HTMLAnchorElement(tag,attrs) : new Element(tag,attrs);
      stack.at(-1).append(node);
      if(!['input','br','hr','img','meta','link'].includes(tag)) stack.push(node);
    } else stack.at(-1).append(new TextNode(decode(token)));
  }
}
function reset(html,url) {
  contexts.clear(); location.href=url;
  document={title:'Synthetic fixture',readyState:'complete',querySelectorAll(s){return flatten(this.body).filter(el=>matches(el,s));},querySelector(s){return this.querySelectorAll(s)[0]||null;}};
  document.body=new Element('body'); document.body.innerHTML=html;
}
reset('<main>Fixture</main>',location.href);
function contextFor(id) {
  if(!contexts.has(id)) contexts.set(id,vm.createContext({document,location,Element,HTMLAnchorElement,HTMLInputElement:Element,
    Node:{DOCUMENT_POSITION_FOLLOWING:4},URL,TextEncoder,TextDecoder,Uint8Array,AbortController,fetch,btoa,setTimeout,clearTimeout,
    getComputedStyle(el) {const style={display:'block',visibility:'visible',opacity:'1',contentVisibility:'visible'};
      for(const part of (el.getAttribute('style')||'').split(';')) {const [key,value]=part.split(':').map(x=>x.trim());if(key) style[key.replace(/-([a-z])/g,(_,s)=>s.toUpperCase())]=value;}
      return style;},
    Event:class {},InputEvent:class {}}));
  return contexts.get(id);
}
async function call(method,params) {
  if(method==='Page.getFrameTree') return {frameTree:{frame:{id:'main'}}};
  if(method==='Page.createIsolatedWorld') return {executionContextId:2};
  if(method==='Page.navigate') {const html=params.url==='about:blank' ? '' : await (await fetch(params.url)).text();reset(html,params.url);return {frameId:'main'};}
  if(method==='Runtime.evaluate') {const value=await vm.runInContext(params.expression,contextFor(params.contextId||1));return {result:{value}};}
  return {};
}
readline.createInterface({input:process.stdin}).on('line',async line=>{
  try {const {method,params}=JSON.parse(line);process.stdout.write(JSON.stringify(await call(method,params))+'\n');}
  catch(error) {process.stdout.write(JSON.stringify({exceptionDetails:{text:String(error)}})+'\n');}
});
