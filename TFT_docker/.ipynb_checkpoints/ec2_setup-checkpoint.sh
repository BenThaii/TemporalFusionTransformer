#! /bin/bash

pip install awscli --upgrade

aws configure set default.region us-east-1

SIZE=${1:-40}
INSTANCEID=$(curl http://169.254.169.254/latest/meta-data//instance-id)
VOLUMEID=$(aws ec2 describe-instances \
  --instance-id $INSTANCEID \
  --query "Reservations[0].Instances[0].BlockDeviceMappings[0].Ebs.VolumeId" \
  --output text)

aws ec2 modify-volume --volume-id $VOLUMEID --size $SIZE
while [ \
  "$(aws ec2 describe-volumes-modifications \
    --volume-id $VOLUMEID \
    --filters Name=modification-state,Values="optimizing","completed" \
    --query "length(VolumesModifications)"\
    --output text)" != "1" ]; do
sleep 1
done

if [ $(readlink -f /dev/xvda) = "/dev/xvda" ]
then
  sudo growpart /dev/xvda 1
  sudo xfs_growfs /dev/xvda1
else
  sudo growpart /dev/nvme0n1 1
  sudo xfs_growfs /dev/nvme0n1p1
fi

