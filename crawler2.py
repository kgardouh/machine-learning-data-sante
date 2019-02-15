import requests
from bs4 import BeautifulSoup
import string
import re

def chomp(s):
    if len(s):
        lines = s.splitlines(True)
        last = lines.pop()
        return ''.join(lines + last.splitlines())
    else:
        return ''

def scrap(root):
	url = root
	code = requests.get(url)
	plain = code.text
	print(url)	
	s = BeautifulSoup(plain, "html.parser")
	data = ""
	#print(s)
	for link in s.findAll('div', {'class':'ipsType_normal ipsType_richText ipsContained'}):
		data = link.find('p')
		if data is not None:
			d = data.get_text()
			#print(d)
			return d



listroots = ["f665-membres.html", "f189-actu-et-débats.html", "f468-culture.html","f641-loisirs.html","f74-informatique.html","f670-jeux-vidéo.html","f119-sports.html","f669-asile.html","f585-forumfr.html"]

for item in listroots:
	for x in range(0,80):
		url = "https://www.forumfr.com/"+item+"?page="+str(x)
		#print("//+"+url)
		c = requests.get(url)
		p = c.text
		soup = BeautifulSoup(p, "html.parser")
		for link in soup.findAll('li', {'class':'ipsDataItem ipsDataItem_responsivePhoto'}):
			l =  link.find('a', {'class':''}).get('href')
			t = link.find('li', {'class':'ipsPagination_page'})

			print("//++++++++++++++"+l)
			#if not t is None:
			with open('./corpus/data-sante'+str(item)+"-"+str(x)+'.txt', 'a') as the_file:
				if str(scrap(l)):
					st = str(scrap(l))
					temp1 = re.sub(r'[^\w\s_]+', '', st).strip()
					temp1 = chomp(temp1)
					if temp1!='':
						#st = st.find('div', {'class':''}).text
						the_file.write(temp1)
						the_file.write('.  	0\n')



	
