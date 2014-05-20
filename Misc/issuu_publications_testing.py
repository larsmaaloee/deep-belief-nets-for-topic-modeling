__author__ = 'larsmaaloee'


import urllib2
import json
import dbn_testing
from numpy import *
from DataPreparation.data_processing import get_all_document_names
from heapq import nsmallest
import serialization as s


def getDocumentURL(documentId):
    try:
        print 'DOCID: ', documentId
        url = 'http://api.issuu.com/query?action=issuu.document.get_user_doc' \
              '&format=json&documentId=%s&documentUsername=&name=' %documentId
        data = json.loads(urllib2.urlopen(url).read())

        if data["rsp"]["stat"] == 'ok':
            docData = data["rsp"]["_content"]["document"]
        try:
            documentName = docData["name"]
            documentOwner = docData["username"]
        except KeyError:
            return 'INTERNAL SERVER ERROR'
        return 'http://issuu.com/%s/docs/%s' % (documentOwner, documentName)
    except Exception, ex:
        print ex
        return 'http://issuu.com/viewer?documentId=' + documentId

def createHTMLReport(sorted_list, readhist, filename='FINAL4'):
    """
    Create a HTML site with the sugested articels
    ============================================
    """
    f0 = open(filename + '.htm', 'w')
    f0.write('<html><body>'
             '<br/>'
             '<p><h1>Similar publications</h1></p>'
             '<br/>'
             '<table border="0">')
    f0.write("<tr>")
    count = 0
    for i in sorted_list:
        f0.write('<td><a href="'+ getDocumentURL(i) + '">'
                 '<img src="http://image.issuu.com/' + i + '/jpg/page_1_thumb_large.jpg" title="'
                 + str(0.0) + '" height="200" width="150"/></a></td>')
        count += 1
        if count == 5:
            f0.write("</tr>")
            count = 0
    f0.write("</table>")
    f0.write('<br/>'
             '<p><h1>Query publication</h1></p>'
             '<br/>'
             '<table border="0">')
    f0.write("<tr>")
    count = 0
    for i in readhist:
        f0.write('<td><img src="http://image.issuu.com/' + i + '/jpg/page_1_thumb_large.jpg" height="200" width="150"/></td>')
        count += 1
        if count == 5:
            f0.write("</tr>")
            count = 0
    f0.write("</table></body></html>")
    f0.close()

def compute_similarity_query(query_doc_id,neighbors,binary_output):
    testing = dbn_testing.DBNTesting(testing = True,image_data=False,binary_output=binary_output)
    output_data = testing.output_data
    class_indices = testing.class_indices
    try:
        doc_names = s.load(open("output/doc_names.p","rb"))
    except:
        doc_names = [dn.split("/")[-1].replace(".p","") for dn in get_all_document_names(training=False)]
        s.dump(doc_names,open("output/doc_names.p","wb"))
    doc_idx = where(array(doc_names) == query_doc_id)[0][0]


    o1 = output_data[doc_idx]
    # Compute distances between o1 and remaining outputs
    distances = []
    for idx2 in range(len(output_data)):
        o2 = output_data[idx2]
        if doc_idx == idx2:
            distances.append(Inf)
            continue
        if binary_output:
            distances.append(dbn_testing.hamming_distance(o1, o2))
        else:
            distances.append(dbn_testing.distance(o1, o2))

    # Retrieve the indices of the n smallest values
    minimum_values = nsmallest(neighbors, distances)

    indices = []
    for m in minimum_values:
        i = list(where(array(distances)==m)[0])
        indices += i

    similar_docs = []
    for i in indices:
        similar_docs.append(doc_names[i])


    createHTMLReport(similar_docs,[query_doc_id],'output/output')




if __name__ == '__main__':
    #test -> [[doc_id, scorr]]
    #readhist -> [doc_id, doc_id, doc_id]
    # 'test_new' -> filename to save to.
    #test = [["110815123059-0d932c975d414c2d8645a22fe9f4ffda",0.0]]
    #readhist = ["110815123059-0d932c975d414c2d8645a22fe9f4ffda"]
    #createHTMLReport(test, readhist, 'test_new')

    compute_similarity_query("090811164852-93aa5e99dd3b475b8c7b5228db72953b",15,False)