(window.webpackJsonp=window.webpackJsonp||[]).push([[12],{"2vz6":function(o,e,t){"use strict";var r=t("qKvR"),a=t("q1tI"),n=t("EB9Y"),i=t("T3Tk");e.a=function(o){var e=o.onClick,t=o.tag,l=o.selectedTag,m=o.scrollToCenter,c=Object(a.useContext)(n.a).isDarkMode,d=Object(a.useRef)(null),g=Object(a.useCallback)((function(){m&&m(d),e&&e(t)}),[m,e,d,t]);return Object(a.useEffect)((function(){l===t&&m&&m(d)}),[m,l,d,t]),Object(r.c)("button",{ref:d,css:Object(r.b)("white-space:nowrap;transition:all 300ms cubic-bezier(0.4,0,0.2,1);font-size:1rem;font-weight:700;border-radius:9999px;margin-right:0.5rem;margin-top:0.25rem;margin-bottom:0.25rem;padding-top:0.25rem;padding-bottom:0.25rem;padding-left:0.75rem;padding-right:0.75rem; ",l===t?"color":"background-color",":",c?"#2d3748":"#edf2f7",";",l===t?"background-color":"color",":",c?i.darkModeColor.textColor1:i.whiteModeColor.textColor1,";"),onClick:g},t)}},"3XHS":function(o,e,t){"use strict";t.r(e);var r=t("wTIg"),a=t("q1tI"),n=t.n(a),i=t("60Qx"),l=t("7oih"),m=t("luWv"),c=(t("pOn1"),t("qKvR"));var d=Object(r.a)("div",{target:"enx0fvl0"})({name:"1bdwg0l",styles:"width:100%;max-width:768px;margin-left:auto;margin-right:auto;"}),g={name:"w27u98",styles:"margin-top:1rem;padding-left:1rem;padding-right:1rem;"},s={name:"1abxlfd",styles:"font-size:2.25rem;font-weight:700;margin-bottom:1rem;@media (min-width: 768px){font-size:3rem;}"},b={name:"cpn9zx",styles:"font-size:1rem;margin-bottom:1rem;"},h={name:"w27u98",styles:"margin-top:1rem;padding-left:1rem;padding-right:1rem;"};e.default=function(o){var e=o.data.allMarkdownRemark.edges,t=Object(a.useState)(void 0),r=(t[0],t[1],e.filter((function(o){return"Resume"===o.node.frontmatter.title})).map((function(o){return o.node}))[0]);return Object(c.c)(n.a.Fragment,null,Object(c.c)(l.a,null,Object(c.c)("div",{css:g,className:"blog-post-container"},Object(c.c)("div",{className:"blog-post"},Object(c.c)(d,null,Object(c.c)("h1",{className:"blog-title",css:s},"Portfolio"),Object(c.c)("h2",{className:"blog-date",css:b},"September 25th 2022"),Object(c.c)(m.a,{color:!0})))),Object(c.c)("div",{css:h,className:"blog-content"},Object(c.c)(d,null,Object(c.c)(i.a,{html:r.html})))))}},"60Qx":function(o,e,t){"use strict";var r=t("qKvR"),a=t("q1tI"),n=t("EB9Y"),i=t("T3Tk");e.a=function(o){var e=o.html,t=Object(a.useContext)(n.a).isDarkMode,l=Object(r.b)("font-size:1rem;word-break:break-word;h1 > a > svg,h2 > a > svg,h3 > a > svg,h4 > a > svg,h5 > a > svg,h6 > a > svg{fill:",t?"#fff":"#000",";}h1,h2{font-size:1.25rem;font-weight:600;margin-top:1.5rem;margin-bottom:1.5rem;}h3,h4,h5,h6{font-size:1.125rem;margin-top:1.5rem;margin-bottom:1.5rem;font-weight:600;}@media (min-width:640px){h1,h2{font-size:1.5rem;}h3,h4,h5,h6{font-size:1.25rem;}}a{color:",t?i.darkModeColor.textColor1:i.whiteModeColor.textColor1,";}a:hover{text-decoration:underline;}p{margin:0.3rem;margin-top:0.75rem;margin-bottom:0.75rem;}ul,ol{margin:0.3rem;margin-left:2rem;}li > p,li > ul,li > ol{margin-bottom:0;}ol{list-style-type:decimal;}ul{list-style-type:disc;}blockquote{padding:0.5rem;background-color:",t?"#333":"#eee",";margin:0.3rem;margin-top:0.5rem;margin-bottom:0.5rem;border-left-width:4px;border-color:",t?i.darkModeColor.mainColor2:i.whiteModeColor.mainColor2,";}blockquote > p{margin:0.5rem;}blockquote > h1,blockquote > h2,blockquote > h3,blockquote > h4,blockquote > h5{margin-top:0.5rem;margin-bottom:0.5rem;}td,th{padding-left:0.5rem;padding-right:0.5rem;padding-top:0.25rem;padding-bottom:0.25rem;border-width:1px;border-color:",t?i.darkModeColor.mainColor2:i.whiteModeColor.mainColor2,";}tr:nth-of-type(even){background-color:",t?"#333":"#eee",";}th{background-color:",t?"#333":"#eee",";}table{margin-bottom:1.5rem;display:block;max-width:fit-content;margin:0 auto;overflow-x:auto;white-space:nowrap;}p > code,li > code{padding-top:0.1rem;padding-bottom:0.1rem;padding-right:0.25rem;padding-left:0.25rem;border-radius:0.25rem;color:",t?i.darkModeColor.textColor1:i.whiteModeColor.textColor1,";background-color:",t?"#333":"#eee",";white-space:pre-line;}pre.grvsc-container{margin-top:24px;margin-bottom:24px;}hr{margin-top:24px;margin-bottom:24px;height:2px;border:none;background:linear-gradient( 270deg,",t?i.darkModeColor.mainColor1+","+i.darkModeColor.mainColor2+","+i.darkModeColor.mainColor3:i.whiteModeColor.mainColor1+","+i.whiteModeColor.mainColor2+","+i.whiteModeColor.mainColor3," );}");return Object(r.c)("div",null,Object(r.c)("div",{css:l,className:"markdown",dangerouslySetInnerHTML:{__html:e}}))}},luWv:function(o,e,t){"use strict";var r=t("qKvR"),a=t("q1tI"),n=t.n(a),i=t("EB9Y"),l=t("T3Tk");e.a=function(o){var e=o.vertical,t=o.color,m=o.margin,c=o.fat,d=Object(a.useContext)(i.a).isDarkMode;return Object(r.c)(n.a.Fragment,null,Object(r.c)("div",{css:e?[{height:"100%",display:"flex",justifyContent:"center"},m&&{marginTop:"0.5rem",marginBottom:"0.5rem"}]:[{display:"flex",justifyContent:"center"},m&&{marginLeft:"0.5rem",marginRight:"0.5rem"}]},Object(r.c)("div",{css:Object(r.b)([{borderRadius:"9999px"},c?e?{height:"100%",width:"0.25rem",marginTop:"auto",marginBottom:"auto"}:{width:"100%",height:"0.25rem"}:e?{height:"100%",width:"1px",marginTop:"auto",marginBottom:"auto"}:{width:"100%",height:"1px"},d?{"--bg-opacity":"1",backgroundColor:"rgba(45, 55, 72, var(--bg-opacity))"}:{"--bg-opacity":"1",backgroundColor:"rgba(247, 250, 252, var(--bg-opacity))"},t&&Object(r.b)("background:linear-gradient( ",e?"180":"270","deg,",d?l.darkModeColor.mainColor1+","+l.darkModeColor.mainColor2+","+l.darkModeColor.mainColor3:l.whiteModeColor.mainColor1+","+l.whiteModeColor.mainColor2+","+l.whiteModeColor.mainColor3," );")],"")})))}},pOn1:function(o,e,t){"use strict";t("q1tI");var r=t("2vz6"),a=t("qKvR");e.a=function(o){var e=o.tags,t=o.onClick,n=o.tag,i=o.scrollToCenter;return e.map((function(o,e){return Object(a.c)(r.a,{tag:o,selectedTag:n,scrollToCenter:i,key:"tag_"+e,onClick:t})}))}}}]);
//# sourceMappingURL=component---src-pages-about-js-3c0cba9242c81692c359.js.map