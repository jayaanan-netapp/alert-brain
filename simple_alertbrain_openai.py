import sys
import openai



sys.path.append('/x/eng/bbrtp/daemon/DOT/dev/ok/tools/openai/')
sys.path.append('/usr/local/lib/python3.10/dist-packages/')
import netapp_openai
import re
import mysql.connector
from simple_mysql_operations import write_alerts_data_into_db, read_all_alerts_from_db

if __name__ == '__main__':
    #First step is to wrtie alerts data into db.
    write_alerts_data_into_db()

    #read all alerts from db

    # Initialize the OpenAI infra for use inside NetApp
    netapp_openai.init()
    print("Alert brain is starting:")


    for alert in read_all_alerts_from_db():
        response = openai.Completion.create(
            engine="gpt-35-turbo-0301",
            model="text-davinci-003",
            prompt="Alert brain to categorize alerts into different categories. We have different five categories named Connector, Performance, Costing, SaaSInfra and  Miscellaneous. Alert should be restricted to these categories.  \n\n" +
            'Alert=  Alert: Connection lost to connector "webserver-1" due to network failure.\n' +
            'Category= Connector\n\n' +
            'Alert=  Alert: Connection failure detected: unable to establish TCP connection to host "database".\n' +
            'Category= Connector\n\n' +

            'Alert=  Alert: Configuration drift detected in SaaS environment; investigate and align configurations for consistency.\n' +
            'Category= SaaSInfra\n\n' +
            'Alert=  Inconsistent service availability across SaaS regions; investigate and ensure equal availability across regions.\n' +
            'Category= SaaSInfra\n\n'+

            'Alert=  ${alert[2]}\n' +
            'Category= ${alert[1]}\n\n',
            #gstop= ["\n", "Alert=", "Category="],
            max_tokens=200,
            temperature=0.5
        )
        print(response.choices[0].text)
        match = re.search(r'Category= (\w+)', response.choices[0].text)

        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="AlertBrain@1234"
        )


        if match:
            category = match.group(1)
            print(category)
            cursor = conn.cursor();
            sql = "UPDATE alert_brain.alert_mapping SET gpt_category = %s WHERE id = %s"
            val = (category, alert[0])

            cursor.execute(sql, val)
            conn.commit()
            cursor.close()
            conn.close()
        else:
            print('no category found')
            cursor = conn.cursor();
            sql = "UPDATE alert_brain.alert_mapping SET gpt_category = %s WHERE id = %s"
            val = ("miscellaneous", alert[0])

            cursor.execute(sql, val)
            conn.commit()
            cursor.close()
            conn.close()
